# Decision Record

<!-- Append-only. Record significant decisions with reasoning. -->

## Pre-tournament stats only (no in-tournament data)
**Date**: pre-2026-03-03
**Context**: Choosing which stat variants to use as features
**Decision**: Pre-tournament stats only (`Pre-Tournament.AdjOE`, etc.)
**Reasoning**: Using stats that include tournament game results would be data leakage — the model would be "seeing" outcomes it's trying to predict.
**Revisit if**: Never — this is a fundamental correctness constraint.

## Barttorvik Data Source and Access Method
**Date**: 2026-03-03
**Context**: KenPom API costs $95/yr, Kaggle datasets may lag. Need current-season pre-tournament stats without being blocked by anti-scraping measures.
**Decision**: Full switch to barttorvik.com via manual browser download using the `team-tables_each.php?csv=1` URL pattern. Use native Barttorvik names for features.
**Reasoning**: Free, comparable data quality (Adjusted efficiency, four factors, tempo, shooting splits). Automated scraping is blocked by Cloudflare verification, but manual downloads work reliably for a once-a-season pull. Covers 2008-2026. Net gain of features over KenPom.
**Revisit if**: Barttorvik adds an official API or if model performance degrades.

## Old KenPom baseline numbers were inflated
**Date**: 2026-03-03
**Context**: Barttorvik retrain showed 0.564 log-loss / 70.3% accuracy vs. old KenPom baseline of 0.364 / 83.3%
**Decision**: Accepted that the old KenPom baseline was likely inflated (likely from data leakage). 
**Reasoning**: 70% accuracy aligns with published benchmarks for NCAA prediction. The model is learning real basketball signal.

## Ensemble weights: XGBoost zeroed out
**Date**: 2026-03-03
**Context**: ENSEMBLE_WEIGHTS needed optimization.
**Decision**: Optimized via scipy Nelder-Mead. Result: [0.5139, 0.0, 0.4861] (LogReg, XGB, RF). XGBoost gets zero weight.
**Reasoning**: XGBoost had the worst LOSO log-loss (0.6063). Optimal ensemble is a blend of log-reg and random forest, yielding a log-loss of 0.5608.

## Matchup predictions use calibrated ensemble model
**Date**: 2026-03-03
**Context**: Bracket engine probability calculations needed the most accurate model.
**Decision**: Standardize on `win_prob_a_ensemble` with optimized weights and fit isotonic regression calibrators on LOSO OOF predictions.
**Reasoning**: Ensemble is the best model. Isotonic regression calibration improved ensemble log-loss to 0.5515. Well-calibrated probabilities are critical for simulation paths. 

## Architectural Pivot to Decoupled Monte Carlo and Strategy Engine
**Date**: 2026-03-16
**Context**: The previous approach (Analytical DP using Champion x Temperature stratified sampling for diversity) was difficult to manage for public-pool EV optimization because it optimized globally instead of reacting to public ownership on an individual pick scale. It conflated probability estimation with pick decision-making.
**Decision**: Replaced the DP engine with a clean decoupling: `src/simulate.py` handles Monte Carlo simulation to generate per-round reach probabilities, and `src/bracket_gen.py` overlays pool-specific logic ("Chalk" vs. "Value/Contrarian").
**Reasoning**: Decoupling probability generation from pick compilation dramatically simplifies the logic and enables explicit pool-type targeting. Ownership-adjusted EV computation (`p_champ / public_ownership_pct`) requires per-round reach probabilities to be pristine and un-adjusted by picking logic. The old DP approach, temperature-based sampling, and forced focal diversity were scrapped as they proved to be a dead-end for optimizing value.
**Revisit if**: Simulation becomes an execution bottleneck, or Monte-Carlo convergence variance is too high for early-round games.
