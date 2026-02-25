# SPEC: Pod-Level Bracket Diversity + SeedNum Inference Fix

**Branch:** `feature/bracket-simulation`  
**Author:** JARVIS  
**Date:** 2026-02-25  
**Status:** Ready for implementation  

---

## Problem

### 1. Bracket diversity is illusory

The current analytical engine fixes the **Final Four** (round 4 regional winners) as its diversity axis. This produces brackets that differ only in rounds 4–6 — roughly 4 of 63 picks (~6%). Rounds 1 and 2, which account for 32 + 16 = 48 picks and most of the actual scoring variance, are **identical across all 5 brackets** for any given forced regional winner.

Root cause: `trace_picks()` is deterministic given a forced regional winner. If all 5 brackets force "Auburn wins the South," they all produce the same 15 South picks below round 4.

### 2. `SeedNum` is still present in inference-time `feature_cols`

`simulate.py:1030` and `predict.py:126, 229` all do:
```python
feature_cols = config.FEATURES + ["SeedNum"]
```
This was left over after `diff_SeedNum` was removed from the training feature set. Currently harmless — `predict_matchup` silently drops it because `scaler.feature_names_in_` doesn't include `diff_SeedNum` — but it's dead code that creates confusion and a latent bug (if the scaler is ever retrained and SeedNum sneaks back in).

---

## Solution

### Part 1: Fix `feature_cols` (small, do first)

Remove `SeedNum` from every `feature_cols` construction. It should just be `config.FEATURES`.

**Files to change:**
- `src/simulate.py` line ~1030: `feature_cols = config.FEATURES + ["SeedNum"]` → `feature_cols = config.FEATURES`
- `src/predict.py` line ~126: same fix
- `src/predict.py` line ~229: same fix
- `src/predict.py` lines ~60–62: remove the `if col == "SeedNum"` special-case branch in `predict_matchup()` — it's unreachable once `SeedNum` is out of `feature_cols`

Verify nothing breaks by running `python3 src/predict.py --season 2025` and confirming probabilities are unchanged (they should be, since the scaler was already ignoring it).

---

### Part 2: Pod-level diversity

#### Background: the tree structure

Each region's bracket tree has a natural hierarchy:

```
Region root (round 4)
├── Left half (round 3)  ← POD A (teams: 1,16,8,9)   POD B (teams: 5,12,4,13)
└── Right half (round 3) ← POD C (teams: 6,11,3,14)  POD D (teams: 7,10,2,15)
```

Each **pod** is a round-2 node: 4 teams, 2 first-round games, 1 Sweet 16 advancer. There are **16 pods total** (4 regions × 4 pods). Forcing which team wins each pod creates diversity that propagates down through rounds 1 and 2 — exactly where we need it.

#### New function: `enumerate_pod_combos()` in `bracket_engine.py`

Replace `enumerate_final_four_combos()` with `enumerate_pod_combos()`.

**Signature:**
```python
def enumerate_pod_combos(
    region_trees: dict[str, BracketNode],
    global_probs: dict[str, dict[int, float]],
    top_k_per_pod: int = 2,
    min_joint_prob: float = 1e-6,
) -> tuple[list[dict[str, str]], list[float]]:
```

**Returns:** Same structure as `enumerate_final_four_combos` — a list of combo dicts and their joint probabilities — except combo keys are **pod IDs** instead of region names.

**Pod ID format:** `"{region}_{pod_index}"` where `pod_index` is 0–3 within the region (derived from `match_idx` of the round-2 node, or just enumerate in tree-walk order). Example: `"East_0"`, `"East_1"`, `"East_2"`, `"East_3"`.

**Implementation steps:**
1. Walk each region tree. Collect all round-2 nodes (these are the pod roots). Each has `.teams` (4 teams) and can compute `global_probs[team][3]` = probability of reaching round 3 = probability of winning the pod.
2. For each pod, sort teams by `global_probs[team][3]`, take top `top_k_per_pod`.
3. `itertools.product()` across all 16 pods. With `top_k_per_pod=2`: 2^16 = 65,536 combos before filtering.
4. Filter by `joint_prob >= min_joint_prob`. Realistic filter removes ~99% of combos, leaving ~500–2000. If fewer than `n_brackets` survive, relax the floor.
5. Return combos and joint probs.

**Helper to collect pod nodes:**
```python
def collect_pod_nodes(region_tree: BracketNode) -> list[BracketNode]:
    """Return all round-2 nodes in left-to-right tree order."""
    pods = []
    def walk(node):
        if node.round_num == 2:
            pods.append(node)
            return
        if node.left: walk(node.left)
        if node.right: walk(node.right)
    walk(region_tree)
    return pods
```

#### Updated `select_diverse_combos()`

The existing function works on any combo dict — **no changes needed**. It just needs the new combo dicts from `enumerate_pod_combos`. 

The distance function `dist(c1, c2)` currently counts differing regions (max 4). With pod combos the max distance is 16. Update the normalization divisor:
```python
# Old:
d_score = avg_d / 4.0
# New:
d_score = avg_d / len(c1)  # normalize by number of keys (works for both pod and F4 combos)
```

#### Updated `evaluate_combo_and_build()` in `bracket_engine.py`

This is the main surgery. Currently it takes a `combo` of 4 regional winners and traces picks top-down. With pod combos, it needs to force picks at the pod level and let the DP handle everything above.

**New logic:**
1. For each region, get the 4 pod winners from the combo dict.
2. Call `trace_picks()` at the pod node level — force the pod winner through rounds 1 and 2 within that pod. This is the same `trace_picks()` call, just starting at the round-2 node instead of the root.
3. For rounds 3+ (Sweet 16, Elite 8, Final Four, Championship): use the existing DP `best_unforced_team` logic at each level, OR re-run a lightweight DP that respects the forced pod winners.

**Recommended approach for step 3:** After forcing the 4 pod winners per region, the Sweet 16 matchups are now known (2 forced pod winners face each other per half-region). Run a simple greedy EV pass upward from there:
- Round 3: forced pod winner A vs forced pod winner B → pick by EV (`global_probs[t][3] * ROUND_POINTS[3]`)
- Round 4 (regional final): the two R3 winners face off → pick by EV
- Round 5–6 (F4, championship): pick by EV from regional winners

This keeps the picks coherent (no impossible matchups) while letting the early-round diversity drive bracket differences.

**Updated signature:**
```python
def evaluate_pod_combo_and_build(
    combo: dict[str, str],           # pod_id -> team_name
    pod_nodes: dict[str, BracketNode],  # pod_id -> round-2 node
    region_trees: dict[str, BracketNode],  # region -> root node
    root: BracketNode,
    global_probs: dict[str, dict[int, float]],
    region_matchups: dict,
    final_four_matchups: list[tuple[str, str]],
    all_teams_by_name: dict[str, 'Team'],
) -> 'Bracket':
```

#### Updated `run_analytical()` in `bracket_engine.py`

Replace the `enumerate_final_four_combos` / `evaluate_combo_and_build` calls with the new pod-level equivalents:

```python
def run_analytical(
    season, region_matchups, final_four_matchups,
    all_teams_by_name, cache,
    n_brackets=5,
    top_k_per_pod=2,        # was top_k_per_region=4
    diversity_weight=0.5,
) -> tuple[list['Bracket'], dict]:
    root, trees = build_full_tree(region_matchups, final_four_matchups)
    global_probs = compute_all_path_probs_tree(root, cache)
    compute_dp(root, global_probs)

    # NEW: collect pods and enumerate at pod level
    all_pod_nodes = {}
    for region, tree in trees.items():
        for i, pod_node in enumerate(collect_pod_nodes(tree)):
            pod_id = f"{region}_{i}"
            all_pod_nodes[pod_id] = pod_node

    combos, combo_probs = enumerate_pod_combos(trees, global_probs, top_k_per_pod)
    selected_combos = select_diverse_combos(combos, combo_probs, n_brackets, diversity_weight)

    out_brackets = []
    for i, combo in enumerate(selected_combos):
        b = evaluate_pod_combo_and_build(
            combo, all_pod_nodes, trees, root, global_probs,
            region_matchups, final_four_matchups, all_teams_by_name
        )
        b.strategy_name = f"Analytical B{i+1}"
        # Summarize forced picks by region for notes
        for region in trees:
            pod_winners = [combo[f"{region}_{j}"] for j in range(4) if f"{region}_{j}" in combo]
            b.notes.append(f"Forced S16 ({region}): {', '.join(pod_winners)}")
        out_brackets.append(b)

    # ... rest unchanged (upset_premiums, diversity, report)
```

---

## Acceptance Criteria

- [ ] `python3 src/predict.py --season 2025` runs without error and probabilities are unchanged after SeedNum removal
- [ ] `python3 src/simulate.py --no-sim` runs without error
- [ ] 5 brackets are generated
- [ ] Mean pairwise overlap drops below 70% (currently ~86%). Target: 60–75%.
- [ ] Min pairwise overlap drops below 60% (currently ~78%). Target: 45–60%.
- [ ] All 5 brackets have different round-1 picks in at least 2 regions
- [ ] Expected scores remain in range 60–80 (no regression from DP quality)
- [ ] No `diff_SeedNum` anywhere in the feature vector at inference time (verify via `scaler.feature_names_in_`)
- [ ] Commit and push to `feature/bracket-simulation`

---

## What NOT to change

- `compute_dp()` — unchanged, still runs bottom-up on the full tree
- `compute_all_path_probs_tree()` — unchanged
- `trace_picks()` — unchanged, just called at pod node level instead of region root
- `select_diverse_combos()` — only the normalization divisor changes
- Monte Carlo simulation — unchanged
- All scoring logic — unchanged
- `bracket_engine.BracketNode` — unchanged

---

## Tuning Notes

If diversity is still too low after implementation:
- Increase `top_k_per_pod` to 3 (→ 3^16 = 43M combos, needs tighter `min_joint_prob` filter, still tractable)
- Increase `diversity_weight` above 0.5 in `select_diverse_combos`

If expected scores drop significantly (>5 pts):
- The greedy EV pass for rounds 3+ may be picking suboptimally. Consider re-running a mini-DP from the forced Sweet 16 set upward instead of a pure greedy pass.

---

## File Summary

| File | Change type | Notes |
|------|-------------|-------|
| `src/bracket_engine.py` | Major | New functions: `collect_pod_nodes`, `enumerate_pod_combos`, `evaluate_pod_combo_and_build`. Update: `select_diverse_combos` divisor, `run_analytical` wiring |
| `src/simulate.py` | Minor | `feature_cols = config.FEATURES` (remove `+ ["SeedNum"]`) |
| `src/predict.py` | Minor | Same `feature_cols` fix × 2 locations; remove dead `SeedNum` branch in `predict_matchup` |
| `src/run_eval.py` | None | No changes needed |
| `src/feature_engineering.py` | None | Already fixed in prior commit |
