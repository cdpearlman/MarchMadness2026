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

## Ownership key naming convention: internal round names
**Date**: 2026-03-16
**Context**: Ownership JSON keys (`sweet_16`, `elite_eight`) didn't match bracket_gen.py's internal round names (`s16`, `e8`), causing silent fallback to uniform ownership.
**Decision**: Standardized ownership JSON to use internal round names: `r64`, `r32`, `s16`, `e8`, `f4`, `champ`. Updated `parse_ownership.py` and all lookups in `bracket_gen.py`.
**Reasoning**: Single source of truth for round name strings. The bracket generator is the primary consumer, so its naming convention wins.
**Revisit if**: Never — just keep the convention consistent.

## Bracket Engine v1.5: Probabilistic Portfolio Generation
**Date**: 2026-03-16
**Context**: Current bracket generation produces near-identical R64/R32 picks across all bracket types. The deterministic upset threshold approach (p_min=0.40 for R64) filters out most upsets, and when one qualifies, all brackets take the same pick. Explored 12 alternative approaches in brainstorming session.
**Decision**: Replace the deterministic chalk/value/contrarian engine with a combined probabilistic generation + portfolio optimization pipeline. Full spec at `specs/bracket_engine_v1.5.md`. Key design:
1. **Champion sampling**: Weighted random from top X% of championship probability mass (no ownership adjustment — let optimizer handle that)
2. **Temperature-based generation**: Generate ~10K brackets across a temperature spectrum. Each game resolved probabilistically using `p_flip = upset_score^(1/temperature)` where `upset_score = p_underdog * (1 - ownership)`. Only champion path is locked.
3. **Monte Carlo evaluation**: Simulate ~50K full tournament outcomes using raw model probabilities
4. **Greedy portfolio selection**: Score all brackets against all simulations, then greedily select N brackets maximizing E[max score] (submodular optimization)
**Reasoning**: Brainstormed 12 approaches. Eliminated: forced upset quotas (supply-blind), partition-based diversity (blocks valuable shared picks), region-flavored brackets (doesn't match how upsets distribute). The probabilistic approach adapts naturally to each year's landscape — few credible upsets = chalky brackets (correct), many toss-ups = natural diversity. Portfolio optimization is the theoretically correct way to select decorrelated brackets. Temperature replaces discrete bracket "types" (chalk/value/contrarian), and bracket character emerges from optimization rather than being prescribed. Champion sampling was designed to be probability-weighted from a quality-filtered pool, ensuring diversity without wasting brackets on longshots.
**Revisit if**: Temperature tuning proves too empirical, or model calibration issues cause the optimizer to degenerate toward chalk.

## Future: Opponent-Aware Portfolio Optimization (v2+)
**Date**: 2026-03-16
**Context**: v1.5 optimizes E[max score] across our portfolio, but the true pool objective is P(finishing 1st), which requires modeling opponent brackets.
**Decision**: Earmarked for v2+. Would generate synthetic opponent brackets from ownership distributions and optimize P(our best bracket beats all opponents). Also includes generating brackets via pure Monte Carlo sampling (without temperature) and selecting portfolios from that — the theoretical ideal if model calibration is strong enough.
**Reasoning**: Adds significant complexity (opponent simulation, larger scoring matrix). v1.5's ownership-weighted generation is a reasonable proxy — it biases toward contrarian picks without formal opponent modeling.
**Revisit if**: v1.5 results suggest ownership weighting in generation is insufficient, or if we want to formally optimize for pool rank rather than raw score.
