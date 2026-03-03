# Lessons Learned

<!-- Append-only. Record what the team learned the hard way. -->

## pre-2026-03-03 — Model degenerates into "just pick the higher seed"
**What happened**: Prediction model achieves decent accuracy but does so primarily by learning that higher seeds win — which is trivially true and not useful for bracket pools where everyone knows the favorites
**Root cause**: Seed differential is an extremely strong signal that can dominate other features. A model that learns "lower seed number = win" gets ~70% accuracy for free.
**Fix**: Monitor SHAP values to ensure non-seed features (efficiency metrics, shooting, etc.) carry meaningful weight. Accuracy alone is not a valid metric — must check log-loss, AUC-ROC, and SHAP feature rankings.
**Rule going forward**: Always validate that the model is learning *beyond* seed. If Seed is overwhelmingly the #1 SHAP feature, the model needs adjustment (feature weighting, regularization, or feature engineering changes).

## pre-2026-03-03 — Bracket generation produces near-identical brackets
**What happened**: Generated bracket set had very high overlap — most brackets picked the same teams in the same spots, defeating the purpose of generating multiple brackets
**Root cause**: Greedy optimization (and Monte Carlo simulation) naturally converges on the single highest-probability path. Without an explicit diversity mechanism, every bracket looks like "pick the favorite in every game."
**Fix**: Implemented diversity weighting in the analytical DP bracket engine. Added `tune_diversity.py` to find the right balance. Goal: maximize P(at least one bracket scores extremely well), not maximize average bracket score.
**Rule going forward**: Always check bracket overlap metrics after generation. If mean overlap is >80%, diversity weight needs increasing. The objective is ONE top-tier bracket, not a set of identical "safe" brackets.
