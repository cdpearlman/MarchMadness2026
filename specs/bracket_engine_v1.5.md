# Bracket Engine v1.5 — Probabilistic Portfolio Generation

## Overview

Replace the current deterministic bracket generation (chalk/value/contrarian types with hard upset thresholds) with a probabilistic generation + portfolio optimization pipeline. Generate thousands of candidate brackets using temperature-controlled probabilistic game resolution, then select the optimal portfolio of N brackets that maximizes E[max score].

## Motivation

The current system produces near-identical R64/R32 picks across all bracket types because:
- The upset validation gate (`p_min=0.40`, `own_max=0.15` for R64) filters out most upsets
- When an upset does qualify, all bracket types take the same pick
- Diversity only emerges in S16+ through champion/F4 locking and contrarian thresholds

This is a structural problem — heuristic upset thresholds can't adapt to the year's actual probability landscape. A year with few credible upsets produces identical brackets; a year with many doesn't exploit them well.

## Architecture

### Pipeline: 4 Stages

```
[1] Champion Sampling  -->  [2] Bracket Generation  --\
     (weighted)              (temperature-based)        +--->  [4] Portfolio Selection
                             [3] Tournament Simulation --/       (greedy optimization)
                                  (Monte Carlo)
```

Note: Stages 2 and 3 are independent and can run in parallel. Stage 4 requires outputs from both.

### Stage 1: Champion Sampling

**Input**: `reach_probabilities_{season}.csv` (from `simulate.py`)

**Method**:
- Rank all teams by `p_champ` descending
- Define a cumulative probability cutoff (e.g., top 80% of total championship mass)
- Walk down the ranked list until cumulative `p_champ` >= cutoff
- These teams form the **champion candidate pool**
- For each generated bracket, sample a champion from this pool with probability proportional to their `p_champ` within the pool

**Example**: If Duke has `p_champ=0.30`, Michigan `p_champ=0.20`, Kansas `p_champ=0.15`, and cutoff is 80% (cumulative 0.65 doesn't reach 0.80, so include next team, etc.), then within the pool, sampling weights are `p_champ / sum(pool p_champ)`.

**Rationale**:
- Pure probability weighting (no ownership adjustment) — ownership's effect is handled downstream by portfolio selection
- Quality floor prevents wasting brackets on longshot champions
- In dominant years, most brackets naturally get the top champion (correct behavior)
- In open years, champions spread across many teams (also correct)
- Stochastic sampling means champion diversity emerges naturally across thousands of brackets

**Open parameter**: Cumulative probability cutoff threshold. Starting point: **0.80** (top 80% of championship mass). Tunable.

### Stage 2: Bracket Generation (Temperature-Based Probabilistic Flipping)

**Input**: Sampled champion, `prob_cache` (pairwise model predictions), ownership data, bracket structure, temperature parameter

**Method — per bracket**:

1. **Lock champion** through all rounds on their path (R64 through Championship)
2. **Resolve each non-locked game forward, round by round** (R64 → R32 → S16 → E8 → F4 → Championship):
   - Compute `p_underdog` from model's pairwise prediction for the two teams in the matchup
   - If `p_underdog < p_floor` (e.g., 0.20): pick the favorite (no flip regardless of temperature). This prevents truly absurd picks.
   - Otherwise, compute **upset score**: `upset_score = p_underdog * (1 - ownership_underdog)`
   - Compute **flip probability**: `p_flip = upset_score ^ (1 / temperature)`
     - At temperature -> 0: `x^inf -> 0` for x<1, so never flips (pure chalk)
     - At temperature=1: `x^1 = x`, flips at the raw upset_score rate
     - At temperature=2: `x^0.5 = sqrt(x)`, e.g. upset_score=0.25 -> p_flip=0.50 (amplified)
     - At temperature -> inf: `x^0 -> 1`, always flips (max contrarian)
   - Draw `random() < p_flip` → if true, pick underdog; else pick favorite
3. **Matchups cascade**: R32 matchups are determined by R64 winners, S16 by R32 winners, etc. Non-standard matchups (e.g., 12-seed vs 13-seed) use the model's pairwise prediction, which handles any team pair.

**Temperature spectrum**: Generate brackets across a range of temperatures. Proposed distribution across N_total brackets:
- ~30% at low temperature (0.1–0.3) — chalk-leaning
- ~40% at medium temperature (0.5–1.0) — moderate upsets
- ~30% at high temperature (1.5–3.0) — contrarian-leaning

This ensures the candidate pool has genuine structural diversity. The optimizer selects from this diverse pool.

**Key difference from current system**: No discrete bracket "types." No hard upset thresholds. No F4 locking. The only constraint is the locked champion path. Everything else is probabilistic, and bracket character (chalk vs. contrarian) is controlled by temperature.

**Open parameters**:
- `p_floor`: Minimum underdog probability to even consider a flip. Starting point: **0.20**
- Temperature range and distribution across brackets
- Whether ownership should be included in flip probability or left entirely to the optimizer (see Risks section)
- Total number of brackets to generate. Starting point: **10,000**

### Stage 3: Tournament Simulation (Monte Carlo)

**Input**: `prob_cache` (pairwise model predictions), bracket structure

**Method**:
- Simulate K complete tournament outcomes by resolving every game probabilistically using raw model probabilities (no temperature, no ownership — just the model's best estimate)
- For each simulation, record the full set of 63 game winners
- This is essentially what `simulate.py` already does, but we need the **full game-by-game outcomes**, not just reach probabilities

**Output**: K simulated tournament results, each a list of 63 winners

**Open parameter**: Number of simulations K. Starting point: **50,000** (matches current simulate.py)

### Stage 4: Portfolio Selection (Greedy Optimization)

**Input**: N_total candidate brackets from Stage 2, K simulated tournaments from Stage 3

**Method**:

1. **Score matrix**: For each (bracket, simulated_tournament) pair, compute the ESPN standard score:
   - R64 correct = 1 pt, R32 = 2 pts, S16 = 4 pts, E8 = 8 pts, F4 = 16 pts, Champ = 32 pts
   - Result: matrix of shape `(N_total, K)` — each cell is a bracket's score in one simulation

2. **Greedy portfolio construction** (select N_portfolio brackets, e.g., 3):
   - **Step 1**: Pick the bracket with the highest average score across all K simulations (the best individual bracket)
   - **Step 2**: For each remaining bracket, compute: "if I add this bracket to my portfolio, what is the new E[max score across portfolio]?" Pick the one that adds the most marginal gain.
   - **Step 3**: Repeat until portfolio has N_portfolio brackets

3. **Why greedy works**: E[max] over a set is a **submodular function** — adding a new bracket has diminishing marginal returns. Greedy maximization of submodular functions is provably within (1 - 1/e) ≈ 63% of optimal. In practice, it's usually much closer.

**Output**: The selected portfolio of N_portfolio brackets, with metadata (temperature used, champion, upset picks, expected scores, pairwise overlap)

**Open parameter**: Portfolio size N_portfolio. Default: **3** (standard pool entry limit). Should be configurable.

## Scoring Function Discussion

### E[max score] vs. E[max pool rank]

The spec above optimizes **E[max score]** — the expected value of the best-scoring bracket in our portfolio across simulated tournaments. This is a good proxy but not the true objective.

The true objective for pool play is **P(at least one bracket finishes 1st in the pool)**, which requires modeling opponent brackets. Opponent behavior can be approximated from ownership data: the "average opponent bracket" picks each team in each round with probability equal to that team's public ownership.

**For v1.5**: Use E[max score]. It's simpler, doesn't require opponent simulation, and ownership influence is already baked into the generation step (via upset_score weighting). The portfolio optimizer naturally decorrelates our brackets from each other; ownership-weighted generation decorrelates them from the field.

**For v2+ consideration**: Add opponent simulation — generate synthetic opponent brackets from ownership distributions, then optimize P(our best bracket beats all opponents). This is the theoretically complete solution but adds significant complexity.

## Computational Feasibility

| Step | Operation | Estimated Cost |
|------|-----------|---------------|
| Stage 2 | Generate 10K brackets | ~10K × 63 games × RNG = trivial (<1s vectorized) |
| Stage 3 | Simulate 50K tournaments | Already done by simulate.py (~30s) |
| Stage 4 | Score matrix | 10K × 50K = 500M pairs, each scoring 63 picks. Vectorized numpy with broadcasting: ~30-60s |
| Stage 4 | Greedy selection | 3 passes × 10K candidates × 50K sims = ~150M ops per pass. <5s |
| **Total** | | **~1-2 minutes** |

The score matrix is the bottleneck. Can be optimized with:
- Sparse representation (most picks match chalk — only store diffs)
- Batch processing in chunks if memory is tight (10K × 50K matrix of int16 = ~1GB)
- Reducing N_total or K if runtime is an issue

## Migration from Current System

### What stays
- `simulate.py` — reach probability generation (still needed for champion candidate ranking)
- `prob_cache` and `build_probability_cache()` — pairwise prediction infrastructure
- `parse_ownership.py` and ownership data — used in upset_score computation
- ESPN standard scoring constants
- Overlap computation (for reporting, not enforcement)
- `bracket_{season}.json` structure and First Four resolution

### What changes
- `bracket_gen.py` — major rewrite. The `generate_bracket()` function becomes a lightweight probabilistic forward pass. New functions for portfolio scoring and greedy selection.
- Bracket types (chalk/value/contrarian) — removed as explicit categories. Replaced by temperature parameter.
- Upset validation (`validate_upset()`) — removed. Replaced by continuous upset_score + probabilistic flipping.
- Champion locking — kept, but champion is sampled rather than deterministically chosen.
- F4 locking — removed. F4 teams emerge from forward generation.
- Hard overlap threshold (85%) — removed as a constraint. Portfolio optimization handles diversity implicitly.

### What's new
- Temperature-parameterized bracket generation loop
- Full tournament outcome simulation (game-by-game, not just reach probabilities)
- Score matrix computation
- Greedy portfolio selection algorithm
- Champion candidate pool construction and weighted sampling

## Open Decisions

1. **Flip probability function**: The spec uses `upset_score ^ (1/temperature)` (power scaling). Alternatives worth testing:
   - Linear: `min(1, temperature * upset_score)` — simpler but clips at 1 and doesn't have smooth behavior
   - Sigmoid: `sigmoid(logit(upset_score) / temperature)` — smooth but asymptotes at 0.5 for high temperatures instead of 1.0
   - The power formulation has correct boundary behavior at all temperature extremes. Start with this and revisit if the upset frequency distribution is unsatisfactory.

2. **Ownership in generation**: Should ownership be part of the upset_score calculation during generation, or should generation use pure model probabilities and let the optimizer handle differentiation?
   - Pro ownership in generation: biases candidate pool toward contrarian picks, giving optimizer better material
   - Pro pure probabilities: avoids double-counting if we later add opponent modeling
   - **Leaning toward**: include ownership in generation for v1.5, revisit if opponent modeling is added

3. **Temperature distribution**: How many brackets at each temperature level? Uniform across range, or weighted toward moderate temperatures?

4. **Champion cutoff threshold**: 80% of cumulative championship mass is a starting point. Might need tuning — too high includes weak champions, too low concentrates on 1-2 teams.

5. **p_floor for flipping**: 0.20 is proposed. A 15-seed with 10% model probability should never be flipped. But where exactly is the line? Could also scale p_floor by round (lower floor in later rounds where upsets are higher-impact).

6. **simulate.py changes**: Currently outputs reach probabilities. We also need full game-by-game simulated outcomes for Stage 3. Options:
   - Extend simulate.py to output both
   - Add a new simulation function in bracket_gen.py
   - Reuse the same simulation but capture full bracket paths

## Risks

### Model calibration sensitivity
Portfolio optimization trusts that simulated tournament outcomes (Stage 3) reflect reality. If the model is miscalibrated (e.g., systematically overconfident in favorites), the optimizer will over-index on chalk because simulated upsets are too rare. **Mitigation**: isotonic calibration is already applied. Validate calibration curve before running.

### Temperature tuning is empirical
There's no theoretical basis for the "right" temperature range. Too narrow → insufficient diversity in candidate pool → optimizer can't find decorrelated portfolios. Too wide → many wasted brackets at extreme temperatures that will never be selected. **Mitigation**: start with a wide range, observe what temperatures the optimizer actually selects, narrow for future runs.

### Interpretability loss
Current system: "This is the value bracket — it picks Duke because Duke has 25% championship probability but only 8% ownership." New system: "The optimizer selected this bracket because it maximizes expected max score in the portfolio." Harder to explain to a human why specific picks were made. **Mitigation**: post-hoc analysis — report each selected bracket's temperature, upset count, champion value score, and overlap metrics.

### Path cascade in forward generation
An early upset rewrites the entire region's trajectory. A 14-seed beating a 3-seed in R64 might then face a weaker R32 opponent and advance further than their reach probability suggests. This is realistic (it happens in real tournaments), but could produce brackets that look incoherent. **Mitigation**: the p_floor prevents truly absurd upsets, and the optimizer filters out brackets with poor expected scores.

### Memory/compute for score matrix
10K brackets × 50K simulations × int16 = ~1GB. Manageable on most machines but worth monitoring. **Mitigation**: chunk processing, or reduce to 5K brackets × 25K simulations if memory is tight (~250MB).

### Champion lock creates a bottleneck
Locking the champion through all 6 rounds means ~10% of each bracket's games are predetermined. If the locked champion happens to be in a region with many toss-up games, those games lose their diversity potential. **Mitigation**: minor concern — 6 out of 63 games is small, and champion path games are usually the most chalk-favoring anyway (champion tends to be a 1-seed).
