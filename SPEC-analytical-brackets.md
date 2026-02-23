# Spec: Analytical Bracket Generator — Backward Induction + Final Four Enumeration

## Context

The current `simulate.py` generates brackets using Monte Carlo simulation +
temperature-based sampling. Through design discussion we identified this approach
has two fundamental problems:

1. **Probability approximation**: Monte Carlo gives noisy reach probabilities.
   Exact analytical computation is faster and strictly more accurate.

2. **Fake diversity**: Temperature sampling produces brackets that differ by
   random noise. Fixing only the champion changes ~6/63 picks. True diversity
   requires different Final Four assumptions — which forces disagreement in all
   4 regions simultaneously (~51 picks changing).

The replacement architecture:

1. **Analytical path probabilities** — exact P(team X reaches round R) via
   recursive forward computation through the bracket tree
2. **Cumulative multi-round EV** — for each team at each position, compute the
   expected score of picking them *across all future rounds from here*, capturing
   favorable/unfavorable draw effects
3. **Final Four enumeration** — generate diverse brackets by committing to
   different plausible Final Four combinations
4. **Backward induction per region** — given a forced regional winner, solve the
   optimal picks within that region using dynamic programming

The Monte Carlo sim is kept as a reporting/validation tool only. Bracket
generation moves entirely to the analytical engine.

---

## Key Concepts

### Analytical Path Probability

For a 16-team single-elimination region with known matchup probabilities:

```
P(team X wins round 1) = head-to-head P(X beats their R1 opponent)

P(team X wins round 2) = P(X wins R1) × Σ_opponents[
    P(opponent j wins R1) × P(X beats j)
]

P(team X wins round 3) = P(X wins R2) × Σ_opponents[
    P(opponent k wins R2) × P(X beats k)
]
... recurse through all rounds
```

This is exact (no sampling), and runs in O(n²) per round = milliseconds.

### Cumulative Multi-Round EV

For team X starting at round R:

```
CumulativeEV(X, R) = Σ_{r=R}^{max_round} P(X reaches round r) × ROUND_POINTS[r]
```

This is what decides upset value. A 12-seed with a favorable draw (faces a beatable
4-seed in R2, then a weak 1-seed matchup) can have higher CumulativeEV than a
12-seed who faces a buzzsaw road. Pure R1 EV misses this entirely.

### Upset Premium

```
UpsetPremium(X, R) = CumulativeEV(X, R) - CumulativeEV(chalk_opponent, R)
```

Positive = underdog is worth picking across multiple rounds combined.
Used to rank which upsets are worth building a bracket around.

### Backward Induction (Constrained Optimal Bracket)

Given a forced regional winner W, solve the region optimally:

```
For each matchup node, working backwards from the championship:
    - If W must win this matchup, W is forced
    - Otherwise, pick whichever team has higher CumulativeEV from this node
```

This produces the highest-EV bracket consistent with W winning the region.
No sampling needed — fully deterministic given the constraint.

### Final Four Enumeration

To generate N diverse brackets:

1. For each of the 4 regions, compute P(each team wins region) analytically
2. Enumerate top-K regional winners per region (e.g. top 4 per region = 4^4 = 256 combinations)
3. Score each combination by joint probability and CumulativeEV
4. Select N combinations that maximize **pairwise diversity** (measured by
   how many regional winner choices differ) while staying above a probability floor
5. For each selected combination, run backward induction in all 4 regions

Result: N brackets that provably differ in all 4 regions, each internally
optimal given their Final Four commitment.

---

## Architecture

### New file: `src/bracket_engine.py`

All analytical computation lives here. `simulate.py` is kept but demoted to
reporting only (Monte Carlo reach probs for visualization).

#### Functions to implement:

```python
def compute_path_probs(
    region_matchups: list[tuple[Team, Team]],
    cache: MatchupCache,
) -> dict[str, dict[int, float]]:
    """
    Exact analytical computation of P(team reaches each round within a region).
    Returns {team_name: {round_1: p, round_2: p, round_3: p, round_4: p}}
    round_1 = R64, round_4 = regional final (E8)
    """
```

```python
def compute_cumulative_ev(
    team_name: str,
    from_round: int,
    path_probs: dict[str, dict[int, float]],
    global_probs: dict[str, dict[int, float]],  # includes F4 + Champ rounds
) -> float:
    """
    Sum of P(team reaches round r) * ROUND_POINTS[r] for r >= from_round.
    global_probs includes Final Four and Championship probabilities.
    """
```

```python
def compute_upset_premiums(
    region_matchups: list[tuple[Team, Team]],
    path_probs: dict[str, dict[int, float]],
    global_probs: dict[str, dict[int, float]],
    cache: MatchupCache,
    min_seed: int = 5,
) -> list[dict]:
    """
    For each high-seeded team, compute UpsetPremium vs their chalk opponent.
    Returns sorted list of:
    {team, seed, opponent, round, single_round_ev, cumulative_ev, upset_premium,
     road_description}
    road_description: e.g. "beats #5, then likely faces #4 (winnable)"
    """
```

```python
def backward_induction_region(
    region_matchups: list[tuple[Team, Team]],
    forced_winner: Team,
    global_probs: dict[str, dict[int, float]],
    cache: MatchupCache,
) -> list[str]:
    """
    Given forced_winner must win the region, return the optimal list of
    picks for all 15 games in the region using backward induction.
    
    For each matchup: if forced_winner is playing, force them to win.
    For all other matchups: pick the team with higher CumulativeEV.
    Returns ordered list of 15 winner names (8 R1 + 4 R2 + 2 S16 + 1 E8).
    """
```

```python
def enumerate_final_four_combos(
    region_path_probs: dict[str, dict[str, float]],  # region -> {team -> p(wins region)}
    top_k_per_region: int = 4,
    min_prob_floor: float = 0.01,
) -> list[dict[str, str]]:
    """
    Generate plausible Final Four combinations.
    Returns list of {region: team_name} dicts, sorted by joint probability.
    Filters out combinations below min_prob_floor joint probability.
    """
```

```python
def select_diverse_combos(
    combos: list[dict[str, str]],
    combo_probs: list[float],
    n: int,
    diversity_weight: float = 0.5,
) -> list[dict[str, str]]:
    """
    From ranked combos, select N that maximize a blend of:
      - Individual combo probability (quality)
      - Pairwise regional disagreement (diversity)
    
    diversity_weight=0.0 → pick top N by probability (low diversity)
    diversity_weight=1.0 → maximize differences (may pick low-prob combos)
    diversity_weight=0.5 → balanced (recommended default)
    
    Selection algorithm:
      1. Always include the highest-probability combo (Bracket 1)
      2. For each subsequent slot: score remaining combos by
         (1 - diversity_weight) * prob + diversity_weight * avg_distance_to_selected
      3. Pick the highest scorer
    """
```

```python
def build_bracket_from_combo(
    combo: dict[str, str],  # {region: forced_winner_name}
    region_matchups: dict[str, list[tuple[Team, Team]]],
    final_four_matchups: list[tuple[str, str]],
    global_probs: dict[str, dict[int, float]],
    all_teams_by_name: dict[str, Team],
    cache: MatchupCache,
) -> Bracket:
    """
    Build a complete bracket for a given Final Four commitment.
    - Regional picks: backward induction per region
    - Final Four picks: pick highest CumulativeEV winner of each semifinal
    - Championship: pick highest CumulativeEV finalist
    Computes expected_score analytically (sum of P(pick correct) * pts).
    """
```

```python
def run_analytical(
    season: int,
    region_matchups: dict[str, list[tuple[Team, Team]]],
    final_four_matchups: list[tuple[str, str]],
    all_teams_by_name: dict[str, Team],
    cache: MatchupCache,
    n_brackets: int = 5,
    top_k_per_region: int = 4,
    diversity_weight: float = 0.5,
) -> tuple[list[Bracket], dict]:
    """
    Full analytical pipeline. Returns (brackets, analysis_report).
    analysis_report contains: path_probs, upset_premiums, final_four_combos,
    selected_combos, diversity_matrix.
    """
```

---

## Tasks

- [ ] 1. **Create `src/bracket_engine.py`** with all functions above

- [ ] 2. **Implement `compute_path_probs()`**

  Use a recursive/iterative forward pass through the bracket tree.
  The bracket has a fixed structure — for a 16-team region with 8 R1 matchups,
  R2 pairings are [winner(0) vs winner(1), winner(2) vs winner(3), ...] etc.
  Represent the bracket as a binary tree and traverse forward from leaves.

  Test: P(1-seed wins region) should be ~85-95% for strong 1-seeds.
  P(12-seed wins R1) should match the direct head-to-head probability.

- [ ] 3. **Implement `compute_cumulative_ev()` and `compute_upset_premiums()`**

  Upset premium report should be printable and human-readable.
  Include `road_description` — e.g. "If they win R1, they face the winner of
  #4 Maryland vs #13 Grand Canyon — a manageable matchup."
  This is the key diagnostic output for deciding which upsets to back.

- [ ] 4. **Implement `backward_induction_region()`**

  Critical correctness requirement: the forced winner must appear in every
  round from R1 through E8. All other picks in their half of the bracket
  must be consistent (you can't have a team winning R2 who lost R1).
  
  The bracket tree structure must be preserved — see `bracket_2025.json`
  for the canonical ordering (position 0 plays position 1's winner in R2, etc.)

- [ ] 5. **Implement `enumerate_final_four_combos()` and `select_diverse_combos()`**

  The diversity metric between two combos is simply:
  ```
  distance(A, B) = number of regions where A and B chose different winners
  ```
  Range: 0 (identical) to 4 (completely different Final Four).
  Target: selected brackets should have mean distance ≥ 2.0.

- [ ] 6. **Implement `build_bracket_from_combo()` and `run_analytical()`**

  `run_analytical()` is the new top-level entry point, replacing
  `generate_all_brackets()` in simulate.py.

- [ ] 7. **Wire into `simulate.py`**

  In `run()`, after loading models and bracket structure:
  - Call `run_analytical()` → produces brackets and report
  - Keep Monte Carlo sim but run it *after* bracket generation (for reporting only)
  - Print upset premium report before brackets
  - Print diversity matrix after brackets (same format as current)

- [ ] 8. **Add `--no-sim` flag to CLI**

  `python src/simulate.py --season 2025 --no-sim` skips Monte Carlo entirely,
  runs only the analytical engine. Useful when you want fast bracket generation
  without waiting 60s for simulation.

- [ ] 9. **Validate on 2025 data**

  Run with `--season 2025 --n-brackets 5` and verify:
  - Upset premium report correctly identifies Colorado State as a high-value pick
    (favorable R2 draw vs Maryland after beating Memphis)
  - At least 2 of 5 brackets have non-identical Final Fours
  - Mean regional winner distance across bracket pairs ≥ 2.0
  - Bracket 1 expected score ≥ 120 (same or better than current greedy)
  - Runtime without `--no-sim` < 90s total; with `--no-sim` < 5s

---

## Constraints

- DO: Keep `Bracket`, `Team`, `MatchupCache`, and all data loading code unchanged
- DO: Keep Monte Carlo sim (`run_monte_carlo()`, `simulate_tournament_once()`) — 
  it moves to reporting only, not removed
- DO: Keep `data/bracket_2025.json` format — bracket_engine.py uses the same
  `region_matchups` dict already produced by `build_region_matchups_from_file()`
- DO: Keep `predict_matchup()` and the logistic model as the probability source
- DON'T: Change `predict.py`, `models.py`, `data_prep.py`, or `config.py`
- DON'T: Remove the existing sampling-based functions yet — deprecate with a
  comment, remove in a follow-up cleanup PR
- Branch: `feature/bracket-simulation` (continue on existing branch)

---

## Done When

`python3 src/simulate.py --season 2025 --n-brackets 5` produces:

1. **Upset Premium Report** — table of high-seed teams with positive cumulative
   EV premium vs their chalk opponent, including road description
2. **5 brackets** — each labeled with their Final Four combo and strategy
3. **Diversity matrix** — mean distance ≥ 2.0 across bracket pairs
4. **No crashes** — clean exit, all outputs saved to processed/

With `--no-sim` flag, runtime < 5 seconds.

---

## Rules & Tips

- The bracket tree structure is: for 8 R1 pairs [0..7], R2 pairs are
  [(0,1), (2,3), (4,5), (6,7)], S16 pairs are [(0,1), (2,3)], E8 is (0,1).
  Preserve this indexing throughout backward induction.
  
- `compute_path_probs()` must account for all possible opponents at each round,
  not just the most likely one. Example: P(X wins R2) depends on P(every possible
  R2 opponent survived R1) × P(X beats each of those opponents). Summing over
  all possible opponents is what makes this exact rather than approximate.

- The Final Four semifinal pairings from `bracket_2025.json` are
  `[["South","West"], ["Midwest","East"]]` — South winner faces West winner,
  Midwest winner faces East winner. Championship is between those two.

- `select_diverse_combos()` greedy selection: start with highest-prob combo,
  then each subsequent pick maximizes (1-w)*prob + w*avg_distance_to_already_selected.
  This is a standard greedy set cover variant — no need for full optimization.

- Expected score validation: the analytical expected score for Bracket 1 should
  be within 2 points of what the Monte Carlo run produces. If it's off by more,
  the path probability computation has a bug.

- For `--no-sim`, the upset premium report is still produced (it uses analytical
  probs, not Monte Carlo). Only `run_monte_carlo()` is skipped.
