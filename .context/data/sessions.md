# Session Log

<!-- Append-only. Add a new entry after each substantive work session. -->

## 2026-03-03 — Bootstrap
**Area**: Project setup
**Work done**: Ran ContextKit bootstrap interview, generated memory system, documented data leakage constraints (seed bias checking, pre-tournament metrics only).

## 2026-03-03 — Barttorvik Data Source Switch
**Area**: Data pipeline / data sourcing
**Work done**: 
- Evaluated data sources and chose barttorvik.com as best free option.
- Discovered automated scraping is blocked by Cloudflare; chose manual browser downloads of `team-tables_each.php?csv=1` (covers 40 columns including height/experience).
- Implemented `ingest_barttorvik.py` mapping to native Barttorvik column names (23 features + SeedNum + derived adj_em). 
- Updated `src/config.py` handling and `team_matching` overrides.

## Previous Attempts: Analytical DP Bracket Engine and Diversity Sampling (Scrapped)
**Area**: Bracket engine pipeline
**Summary**: Initial attempts at bracket optimization used an analytical Dynamic Programming (DP) engine over Monte Carlo, generating brackets by varying "temperature" thresholds across paths combined with forced regional winners. 
**Outcome**: Scrapped. While it created structurally diverse brackets, the approach proved to be a dead-end for public pool optimization because it optimized probabilities globally instead of reacting to public ownership on an individual pick level. 

## 2026-03-03 — Pipeline Audit & Model Improvements
**Area**: Full pipeline evaluation, model layer
**Work done**:
- Optimized ensemble weights via scipy.optimize (Result: [0.5139 LogReg, 0.0 XGBoost, 0.4861 RF]).
- Switched default matchup predictions (e.g. MatchupCache) to use the ensemble model instead of logistic regression.
- Added isotonic probability calibration on LOSO OOF predictions, improving ensemble log-loss to 0.5515.
- Built backtesting framework (`src/backtest.py`) for game-level LOSO comparison against seed-only baseline.
**Decisions made**: XGBoost zeroed out; ensemble model with calibration handles all baseline win probability generation.

## 2026-03-16 — Monte Carlo Simulation & Bracket Strategy Refactor
**Area**: Bracket simulation and picking strategy
**Work done**: 
- Replaced the old deterministic bracket engine with a true Monte Carlo simulator (`src/simulate.py`) that propagates calibrated per-game win probabilities through the 63-game tournament tree.
- Generated path probabilities (`data/processed/reach_probabilities_2026.csv`) and calculated expected standard ESPN pool scores.
- Implemented `src/bracket_gen.py` applying two main strategies: "Model Chalk" (pure EV, for small pools) and "Value Champion/Contrarian" (ownership-adjusted EV, for large pools).
- Removed the old diversity codebase since the new simulation properly isolates the variance without contrived sub-sampling.
- Created `data/ownership_2026.json` structure to hold public pick percentages.
- Merged the `jarvis/bracket-strategy` branch into `main` and resolved merge conflicts.
**Decisions made**: 
- Pivot to decoupled architecture: `simulate.py` (pure probability math) -> `bracket_gen.py` (strategy overlay).
- Incorporate public ownership fractions as a denominator for champion logic to maximize EV in large public pools.
**Open threads**:
- Await real 2026 Selection Sunday bracket assignments (`data/bracket_2026.json`).
- Gather real ESPN/Yahoo public ownership metrics leading into Thursday.

## 2026-03-16 — Bracket Generator Bug Fixes
**Area**: Bracket generation (`src/bracket_gen.py`)
**Work done**:
- Fixed First Four resolution: was arbitrarily picking first team from `TBD_TeamA_TeamB` string split; now uses model win probabilities via `resolve_first_four()`.
- Fixed value bracket Final Four selection: was picking all-value for non-champion regions; now properly implements 2-chalk + 1-value by finding the region with the best value differential.
- Fixed overlap calculation: was using set intersection (position-unaware) against only the last bracket; now uses position-aware zip comparison against ALL previous brackets (reports max overlap).
- Added pairwise overlap validation: warns if any bracket pair exceeds 85% overlap threshold.
- Replaced remaining Unicode box-drawing characters (━, ─, →) with ASCII equivalents (=, -, ->).
- Extracted `compute_bracket_overlap()` as a standalone function for reuse.

## 2026-03-16 — Real Ownership Data + Bug Fixes Round 2
**Area**: Ownership data pipeline, bracket generation bugs
**Work done**:
- Scraped Yahoo pick distribution data from `ownership_*.md` files (manually copied from Yahoo Fantasy) and built `parse_ownership.py` to parse and map team names (Yahoo -> bracket names).
- Populated `data/ownership_2026.json` with real percentages across all 6 rounds (68 teams each), replacing placeholder data.
- Fixed ownership key mismatch: code used internal round names (`s16`, `e8`) but JSON had descriptive names (`sweet_16`, `elite_eight`). Standardized JSON keys to match internal names (`r64`, `r32`, `s16`, `e8`, `f4`, `champ`).
- Fixed play-in loser bug: `reach_probs` CSV included eliminated First Four losers. Texas (play-in loser) had 0.01% F4 ownership, inflating its value score to 34.6x and landing it in the Final Four. Added early filtering of eliminated teams from reach_probs in `generate_bracket()`.
- Verified all ownership lookups now use real Yahoo data (spot-checked Kansas 7.93%, Vanderbilt 10.57%, Arkansas 10.86%).
**Current bracket output (with real ownership)**:
- B1 (chalk): Michigan champ, Duke/Arizona/Illinois F4. EV=88.4
- B2 (value): Texas Tech champ (9.9x value), Louisville/Arizona/Illinois F4. EV=67.9
- B3 (contrarian): Illinois champ (7.6x value), Louisville/N.C. State/Texas Tech F4. EV=61.9
**Open threads**:
- B2 vs B3 overlap is 88% (exceeds 85% threshold). Root cause: R64 is 32/32 identical across all brackets (always chalk), R32 is 15-16/16. Differentiation only happens at S16+. Need to inject early-round upsets in value/contrarian brackets to break overlap.

## 2026-03-16 — Bracket Engine v1.5 Design Session
**Area**: Bracket generation architecture
**Work done**:
- Brainstormed 12 approaches to solve early-round bracket convergence. Evaluated each against constraints: variable upset supply across years, allowing valuable upsets in multiple brackets, pool EV optimization, and computational feasibility.
- Eliminated: forced upset quotas (#3/#4 — supply-blind), anti-correlation/partitioning (#7/#8 — blocks shared valuable picks), region-flavored brackets (#11 — doesn't match upset distribution reality).
- Selected combined approach: temperature-based probabilistic generation + Monte Carlo portfolio optimization. This merges what were originally separate v1.5 (probabilistic flipping) and v2 (portfolio optimization) ideas into a single architecture.
- Designed 4-stage pipeline: champion sampling -> bracket generation -> tournament simulation -> greedy portfolio selection. Full spec at `specs/bracket_engine_v1.5.md`.
- Key design decisions: (1) Only lock champion, not F4 — F4 diversity emerges from temperature. (2) Champion sampled proportionally from top X% of championship probability mass, not deterministically selected. (3) Bracket "types" (chalk/value/contrarian) replaced by continuous temperature parameter. (4) Greedy submodular optimization for portfolio selection.
- Found and fixed a bug in the spec: original flip probability formula `sigmoid(temperature * logit(upset_score))` was backwards (higher temp = less flipping). Corrected to power scaling `upset_score^(1/temperature)`.
- Earmarked opponent-aware optimization as v2+ (generating synthetic opponent brackets from ownership data to optimize P(1st place) instead of E[max score]).
**Decisions made**: Full architecture documented in `specs/bracket_engine_v1.5.md` and decision record.
**Open threads**:
- Open parameters to tune: p_floor, temperature range/distribution, champion cutoff threshold, ownership's role in generation vs. pure probability
- simulate.py needs extension to output full 63-game outcomes (not just reach probabilities)
- Implementation not yet started
