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

## 2026-03-17 — Bracket Engine v1.5 Implementation + Edge-Clamped Leverage Scoring
**Area**: Bracket generation (`src/bracket_gen.py`, `src/config.py`)
**Work done**:
- Implemented full v1.5 pipeline: champion sampling, temperature-based generation, Monte Carlo evaluation, greedy portfolio selection.
- Fixed 3 output bugs: `final_four` JSON key mapped to wrong slot range (56-60 instead of 60-62), print report labels mismatched, missing "Finals" line.
- Identified fundamental flaw in model-only E[max] scoring: optimizer selects chalk because model is ground truth, undoing generation diversity.
- Analyzed pure 1/ownership leverage scoring — proved mathematically that when field is well-calibrated, E[leverage_value] is equal for ALL teams regardless of quality, making optimizer indifferent between Duke and Penn.
- Implemented edge-clamped leverage scoring: `weight = min(cap, max(1.0, model_reach / ownership))`. Only boosts picks where model is more bullish than the field. Cap at 3.0x prevents extreme long-shot amplification.
- Changed score matrix dtype from uint8 to uint16 (leverage scores can exceed 255). Memory: 750MB -> 1500MB.
- Performance regression: scoring loop ~257s vs ~108s (element-wise multiply vs dot product). Acceptable for one-shot generation.
- Results: champions shifted from over-owned (Duke/Arizona) to model-edge teams (Illinois/Purdue/Houston). Overlap improved slightly (0.86-0.87 vs 0.87-0.89).
**Config changes**: `BRACKET_N_TOTAL` 10K->15K, added `BRACKET_EDGE_CAP = 3.0`.

## 2026-03-17 — Portfolio Sizing for Multi-Platform Entry
**Area**: Bracket generation strategy
**Work done**:
- Analyzed platform entry limits: ESPN (25), Yahoo (10), single-entry contests (Kalshi, bet365, CBS).
- Bumped `BRACKET_N_PORTFOLIO` from 3 to 25 (ESPN max). Greedy portfolio means first N brackets are an optimally-selected N-bracket portfolio — take prefixes for smaller platforms.
- Generated full 25-bracket portfolio. E[max] curve: 93.2 (1 bracket) -> 141.0 (10 brackets, Yahoo) -> 151.2 (25 brackets, ESPN). Diminishing returns flatten after ~10.
- Champion diversity across 25: Duke(5), Michigan(4), Arizona(4), Illinois(3), Purdue(2), Houston(2), Florida(2), Connecticut(2), Michigan St.(1). All 9 pool teams represented.
- Later brackets (#11+) introduce contrarian elements (Vanderbilt finals, Nebraska finals, Louisville F4) for large-pool differentiation.
**Entry plan**: ESPN #1-25, Yahoo #1-10, single-entry contests use #1.
**Open threads**: Results not yet validated against real tournament outcomes (tournament starts 2026-03-20).

## 2026-03-17 — Full Pipeline Re-run + Championship Score Estimates
**Area**: Data refresh, model retraining, bracket generation, scoring research
**Work done**:
- Re-ran full pipeline after user updated Barttorvik CSVs and re-ran download/ingest: team_matching -> data_prep -> models -> simulate -> bracket_gen.
- Fixed Unicode crash in `src/simulate.py` (box-drawing chars `═` and `─` in print statements -> `=` and `-`).
- Updated `ENSEMBLE_WEIGHTS` in config to `[0.5467, 0.0, 0.4533]` based on newly optimized LOSO weights (prev: [0.5139, 0.0, 0.4861]).
- Model performance after refresh: Ensemble log-loss=0.5608, accuracy=70.1%, AUC=0.778. SHAP top features: diff_adj_em (0.724), diff_adj_d (0.292), diff_adj_o (0.156) — non-seed features dominant, no seed bias.
- 50K-sim probability matrix generated. Top contenders: Michigan 14.2%, Duke 14.1%, Purdue 11.5%, Arizona 10.6%, Houston 8.1%, Illinois 8.0%.
- 10-bracket portfolio generated. Champions: Florida(x2), Arizona, Iowa St.(x2), Michigan, Illinois, Purdue, Houston, Michigan St.
- Researched offensive/defensive style and scoring averages for all top-3 seeds. Computed estimated championship game scores using Barttorvik adj_o/adj_d/adj_t with 0.93 championship factor.

**Championship score estimates by bracket:**

| Bracket | Champion | Matchup | Est. Score |
|---------|----------|---------|------------|
| 1 | Florida | Florida vs Purdue | Florida 79, Purdue 77 |
| 2 | Arizona | Houston vs Arizona | Arizona 73, Houston 71 |
| 3 | Iowa St. | Illinois vs Iowa St. | Iowa St. 76, Illinois 74 |
| 4 | Michigan | Duke vs Michigan | Michigan 75, Duke 74 |
| 5 | Illinois | Illinois vs Purdue | Illinois 80, Purdue 79 |
| 6 | Purdue | Houston vs Purdue | Purdue 73, Houston 71 |
| 7 | Houston | Houston vs Virginia | Houston 73, Virginia 68 |
| 8 | Iowa St. | Houston vs Iowa St. | Iowa St. 70, Houston 68 |
| 9 | Michigan St. | Michigan St. vs Arkansas | Michigan St. 80, Arkansas 78 |
| 10 | Florida | Florida vs Arizona | Florida 75, Arizona 73 |

**Style notes**: Houston/Virginia games trend lowest (grind pace, adj_t ~64-66). Purdue/Illinois games trend highest (both adj_o 131+, soft defense). Houston is the most consistent suppressor — all Houston finals estimated in 68-73 range.
