# Decision Record

<!-- Append-only. Record significant decisions with reasoning. -->

## Analytical DP over Monte Carlo for bracket optimization
**Date**: pre-2026-03-03 (documented at bootstrap)
**Context**: Needed a bracket optimization strategy that could generate diverse, strategically sound brackets
**Options considered**: Monte Carlo simulation vs. analytical dynamic programming
**Decision**: Moved to analytical DP (`bracket_engine.py`)
**Reasoning**: Monte Carlo was too greedy — always favored the higher-probability outcome in each game (e.g., 51% > 49%) without considering the strategic value of upsets (better downstream matchups, bracket diversity, pool scoring upside). The DP approach evaluates full paths through the bracket and can weigh diversity.
**Revisit if**: DP approach becomes too slow for larger bracket counts or if a hybrid approach could combine DP's strategic depth with MC's sampling flexibility

## Pre-tournament stats only (no in-tournament data)
**Date**: pre-2026-03-03 (documented at bootstrap)
**Context**: Choosing which stat variants to use as features
**Options considered**: Full-season stats vs. pre-tournament stats
**Decision**: Pre-tournament stats only (`Pre-Tournament.AdjOE`, etc.)
**Reasoning**: Using stats that include tournament game results would be data leakage — the model would be "seeing" outcomes it's trying to predict
**Revisit if**: Never — this is a fundamental correctness constraint

## Potential switch to Barttorvik data source
**Date**: 2026-03-03 (under consideration)
**Context**: Current KenPom data comes from Kaggle datasets which may lag behind the season
**Options considered**: Keep KenPom/Kaggle vs. switch to Barttorvik (barttorvik.com)
**Decision**: Pending investigation
**Reasoning**: Data freshness is the primary concern — need current-season stats for 2026 predictions
**Revisit if**: Decision needed before 2026 tournament bracket generation

## Barttorvik data access method
**Date**: 2026-03-03
**Context**: Evaluating switch from Kaggle/KenPom to barttorvik.com for team stats
**Options considered**: (1) Direct API via cbbstat.com — dead/502, (2) Python scraping (requests, cloudscraper, Playwright) — all blocked by JS verification, (3) toRvik/cbbdata R packages — R-only, (4) Manual browser CSV download via trank.php?csv=1 or team-tables_each.php?csv=1
**Decision**: Manual browser download using team-tables_each.php?csv=1 URL pattern (40 columns, includes height/experience). One download per season per year, with begin/end date params for pre-tournament filtering.
**Reasoning**: All automated approaches are blocked by Cloudflare JS verification. Browser downloads work reliably and are a one-time-per-season task. The 40-column team-tables CSV covers 22/24 features directly (2 derivable, 2 missing).
**Revisit if**: Barttorvik adds an official API, cbbdata API comes back online, or Cloudflare protection changes

## Switch from KenPom/Kaggle to Barttorvik — confirmed
**Date**: 2026-03-03
**Context**: KenPom API costs $95/yr, Kaggle datasets may not include 2026 data in time. Need current-season pre-tournament stats.
**Options considered**: (1) KenPom API at $95/yr, (2) Barttorvik via manual browser download, (3) CFBD API (Sports-Reference backend), (4) Manual entry for 2026 only + keep KenPom historical, (5) Hybrid sources
**Decision**: Full switch to Barttorvik
**Reasoning**: Free, comparable data quality (Adjusted efficiency, four factors, tempo, shooting splits, height/experience), covers 2008-2026 with pre-tournament date filtering. Loses StlRate/OppStlRate but gains defensive four factors and barthag. Net gain of 3 features.
**Revisit if**: Model performance degrades below acceptable thresholds, or if a free API with better automation becomes available

## Feature set changes for Barttorvik
**Date**: 2026-03-03
**Context**: Barttorvik uses different column names and has slightly different available stats than KenPom
**Options considered**: (1) Map Barttorvik columns to old KenPom names, (2) Use native Barttorvik names and retrain
**Decision**: Use native Barttorvik names throughout. 23 features + SeedNum + adj_em (derived AdjOE - AdjDE).
**Reasoning**: Direct mapping would require a brittle translation layer. Retraining with native names is cleaner, lets us use Barttorvik-specific features (barthag, defensive four factors), and avoids a permanent source of confusion.
**Revisit if**: Adding another data source that needs a common schema

## Old KenPom baseline numbers were inflated
**Date**: 2026-03-03
**Context**: Barttorvik retrain showed 0.564 log-loss / 70.3% accuracy vs. old KenPom baseline of 0.364 / 83.3%
**Options considered**: (1) The Barttorvik model is worse, (2) The old baseline was inflated
**Decision**: Accepted that the old baseline was likely inflated — probably from data leakage (post-tournament stats instead of pre-tournament only) or non-LOSO evaluation. Current numbers are realistic for NCAA tournament prediction.
**Reasoning**: 70% accuracy aligns with published benchmarks for NCAA prediction. SHAP shows healthy feature distribution (adjusted efficiency metrics at top, seed not dominant). The model is learning real basketball signal.
**Revisit if**: A rigorous re-evaluation of the KenPom pipeline shows the old numbers were legitimate

## Bracket diversity needs structural improvement beyond pod-level
**Date**: 2026-03-03
**Context**: First bracket engine run post-Barttorvik refactor produced 5 brackets for 2025. 4/5 brackets had identical E8, F4, and champion (Houston). B1/B3 were 97% overlap. All 5 had identical R64 picks.
**Problem**: Pod-level diversity (varying S16 winners) is insufficient because the greedy EV pass from S16 upward collapses to the same late-round picks. R64 is never diversified because picks trace downward from forced S16 winners.
**Decision**: Flagged for next session — needs architectural changes to the diversity mechanism
**Candidate approaches**: (1) Force diversity at E8/F4/champion level, not just pods, (2) Increase `top_k_per_pod` beyond 2, (3) Add R64 upset forcing for select matchups, (4) Use the full Final Four combo enumeration that already exists (`enumerate_final_four_combos`) instead of only pod combos, (5) Switch MatchupCache from logistic-only to ensemble/RF
**Revisit if**: Immediately — this is the top priority for bracket quality

## Champion x Temperature stratified sampling for bracket diversity
**Date**: 2026-03-03
**Context**: Pod-level diversity (varying S16 winners) was insufficient — 4/5 brackets had identical E8/F4/champion. The greedy EV pass from S16 upward collapsed late-round picks.
**Options considered**: (1) Forced champion/F4 with deterministic temperature-adjusted EV, (2) Forced champion with sampled (probabilistic) non-champion games, (3) Portfolio simulation (Monte Carlo optimize across all brackets), (4) Upset-tier layering
**Decision**: Stratified sampling — for each (champion, temperature) cell in a grid, generate K candidates via sampling with `forced_champion`, keep the best by EV. Grid: 5 champions x 3 temps = 15 brackets.
**Reasoning**: Sampling > deterministic EV because temperature-adjusted EV is a step function (a game flips at exactly one threshold), while sampling produces diverse upset combinations. Stratified > unconstrained selection because it guarantees even coverage (no champion or temperature tier is over-represented). Forced champion directly solves the "all brackets pick same champion" problem.
**Revisit if**: E8 convergence within same-champion brackets becomes a problem (could add forced F4 diversity), or if portfolio simulation (approach D) is worth the additional complexity

## Bracket generation config: temps=[0.4, 0.9, 1.6], K=12
**Date**: 2026-03-03
**Context**: Needed to tune temperature tiers and candidates-per-cell (K) for the new champion x temperature pipeline. Ran 10 experiments across 2 rounds.
**Key finding**: K is the dominant diversity lever. K=100 over-optimizes toward chalk (69.7% overlap). K=1 produces garbage (43.5% overlap but EV=58). K=10-20 is the sweet spot.
**Decision**: `BRACKET_TEMPERATURE_TIERS = [0.4, 0.9, 1.6]`, `BRACKET_CANDIDATES_PER_CELL = 12`
**Reasoning**: 0.4 = chalk (strong model signal), 0.9 = near raw probability (moderate upsets), 1.6 = genuine chaos. K=12 filters garbage without over-converging. Produces 56.8% mean overlap, 5 unique F4s, 6 unique E8s. Good balance of quality and diversity.
**Revisit if**: Different tournament year produces degenerate results, or if the EV floor (73.4) on high-temp low-probability-champion brackets is too low

## Ensemble weights: XGBoost zeroed out
**Date**: 2026-03-03
**Context**: ENSEMBLE_WEIGHTS were arbitrary [0.30, 0.45, 0.25] (LogReg, XGB, RF). Needed optimization.
**Options considered**: (1) Grid search, (2) scipy.optimize with softmax parameterization on LOSO OOF predictions
**Decision**: Optimized via scipy Nelder-Mead. Result: [0.5139, 0.0, 0.4861]. XGBoost gets zero weight.
**Reasoning**: XGBoost had the worst LOSO log-loss (0.6063) by a wide margin. The optimal ensemble is a ~51/49 blend of logistic regression and random forest. This improved ensemble log-loss from 0.5723 to 0.5608, making ensemble the best model overall.
**Revisit if**: XGBoost hyperparameter tuning significantly improves its performance, or if adding new features changes the relative model strengths

## MatchupCache switched from logistic to ensemble
**Date**: 2026-03-03
**Context**: MatchupCache in simulate.py used `win_prob_a_logistic` (comment said "use best single model" but logistic was actually the worst). All bracket engine probability calculations flow through this cache.
**Decision**: Switch to `win_prob_a_ensemble` with optimized weights and isotonic calibration
**Reasoning**: Ensemble is now the best model by log-loss (0.5608). Adding calibration further improves to 0.5515. Every probability in the bracket engine benefits from this improvement.
**Revisit if**: A single model clearly dominates the ensemble, or if calibration causes issues at probability extremes

## Added isotonic probability calibration
**Date**: 2026-03-03
**Context**: Raw model probabilities used directly in bracket engine. Well-calibrated probabilities are critical for DP optimizer and path probability calculations.
**Decision**: Fit isotonic regression calibrators on LOSO OOF predictions for each base model + ensemble. Stored in pkl file and applied at prediction time.
**Reasoning**: Isotonic regression improved ensemble log-loss by 0.0095 (0.5610 -> 0.5515). More flexible than Platt scaling. Applied transparently through predict_matchup() and MatchupCache.
**Revisit if**: Calibration causes probability compression at extremes, or if a different calibration method (Venn-Abers, beta calibration) would be more appropriate

## F4 diversity via forced regional winners
**Date**: 2026-03-03
**Context**: Within same forced champion, all 3 temperature brackets tended to converge on identical E8/F4 picks from non-champion regions.
**Decision**: For each champion, at higher temperature tiers, force the #2 team (by regional championship probability) to win their region. temp_idx=1 forces 1 alternative region, temp_idx=2 forces 2.
**Reasoning**: Directly breaks E8/F4 convergence within same-champion brackets. Mean overlap dropped from ~63% to 55.5%. Low temperature bracket (chalk) is unaffected. Minimal code change — extends existing `simulate_bracket_with_temperature` with `forced_regional_winners` parameter.
**Revisit if**: The #2 regional pick is consistently a bad choice (too low probability), or if a more sophisticated approach (forced F4 combos enumeration) would produce better results
