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

## 2026-03-03 — Barttorvik Cloudflare blocks all programmatic access
**What happened**: Spent significant time trying to bypass barttorvik.com's JS browser verification with requests, cloudscraper, and Playwright (headless and non-headless). All failed.
**Root cause**: The site uses a JavaScript form auto-submit verification that requires a real browser session. Even Playwright with anti-detection measures couldn't pass it.
**Fix**: Use manual browser downloads — the CSV URLs work fine when visited by a human.
**Rule going forward**: Don't rabbit-hole on scraping protected sites. Check if manual download is acceptable first — for a one-time-per-season data pull, automation isn't worth the effort.

## 2026-03-03 — UL Monroe TeamID was wrong for years
**What happened**: MANUAL_OVERRIDES in config.py had "ul monroe" mapped to TeamID 1349, which is actually Rice. Should be 1419 (ULM).
**Root cause**: Likely a copy-paste error when original overrides were created. Never caught because the pipeline didn't validate override correctness.
**Fix**: Corrected to 1419 during Barttorvik migration.
**Rule going forward**: When adding manual overrides, verify the TeamID against MTeams.csv. Don't trust existing overrides blindly — spot-check when touching them.

## 2026-03-03 — diff_SeedNum was never being computed
**What happened**: feature_engineering.py only computed differentials for features listed in config.FEATURES. SeedNum was intentionally excluded from FEATURES (to avoid seed bias in the main feature list) but was also never explicitly added to the differential computation. So the model never had access to seed differential as a feature.
**Root cause**: The differential code iterated over `config.FEATURES` and `config.FEATURES_PHYSICAL` only. SeedNum lived separately.
**Fix**: Added explicit handling in `compute_differentials()` to include SeedNum in the differential list.
**Rule going forward**: If a feature exists in the data but isn't in the FEATURES list, check whether it still needs differential computation. The FEATURES list controls model input, but differential computation might need additional columns.

## 2026-03-03 — Emoji characters crash on Windows (cp1252)
**What happened**: `print()` statements with emoji characters (⚠️, ✅) in team_matching.py and data_prep.py threw `UnicodeEncodeError: 'charmap' codec can't encode characters`.
**Root cause**: Windows PowerShell defaults to cp1252 encoding, which can't represent emoji codepoints.
**Fix**: Replaced all emoji with plain ASCII markers: `[!]`, `[OK]`, `[WARN]`, etc.
**Rule going forward**: Never use emoji in print statements. Use plain ASCII markers for console output. This is a Windows compatibility constraint.

## 2026-03-03 — Old KenPom model baseline was likely inflated
**What happened**: Old KenPom model reported 0.364 log-loss, 83.3% accuracy, 0.920 AUC. After switching to Barttorvik with a clean LOSO pipeline, best model achieved 0.564 log-loss, 70.3% accuracy, 0.775 AUC. The gap was too large to explain by feature differences alone.
**Root cause**: Most likely the old evaluation used post-tournament stats (data leakage) or wasn't doing proper LOSO (some training seasons leaked into test folds). 83% accuracy is unrealistically high for pre-tournament-only prediction of March Madness games.
**Fix**: Accepted 70% accuracy as the realistic baseline. This aligns with published benchmarks.
**Rule going forward**: Be skeptical of evaluation metrics. Always verify: (1) stats are truly pre-tournament, (2) LOSO folds are clean, (3) results align with published baselines (~68-72% for NCAA tournament prediction).
