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
