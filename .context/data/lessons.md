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

## 2026-03-03 — Stale closure variable in path probability computation
**What happened**: `compute_all_path_probs_tree()` in `bracket_engine.py` computed near-zero probabilities for all right-side-of-bracket teams at every tree level. Championship probability was 0.0% for ALL 64 teams. Probability sums across teams at each round were far below expected (e.g., R32 sum=12.3 instead of 16, Champ sum=0.0001 instead of 1.0).
**Root cause**: The second for-loop (for right-side teams) used `p_a` in a sum comprehension, but `p_a` was a stale reference to the last value from the FIRST for-loop (over left-side teams). The `t_a` iterator variable in the generator shadowed the outer loop, but `p_a` was NOT re-bound — it always held the probability of the last left-side team (typically a #16 seed at ~0.8%). Fix: replace `p_a` with `node.left.node_probs[t_a.name]`.
**How it was caught**: Verified probability conservation — sums across teams at each round should equal the number of games (32, 16, 8, 4, 2, 1). The massive gap immediately revealed the bug.
**Rule going forward**: When computing path probabilities in tree structures, always verify conservation: the sum of node_probs at each level must equal 1.0 for each subtree. Run this check after any change to the probability computation.

## 2026-03-03 — Bracket JSON names don't match Barttorvik names
**What happened**: Teams like "Ole Miss", "UConn", "Norfolk State" in `bracket_2025.json` had no match in `team_stats.csv` (Barttorvik uses "Mississippi", "Connecticut", "Norfolk St."). The partial matching fallback was too greedy — "alabama state" matched "alabama" because "alabama" is a substring of "alabama state".
**Root cause**: No alias layer between bracket file names and data source names. Partial matching by substring is unreliable when short names are substrings of longer ones.
**Fix**: Added `bracket_aliases` dict for known mismatches. Improved partial matching to prefer closest-length match.
**Rule going forward**: When adding a new season's `bracket_YYYY.json`, test name matching first — run `build_region_matchups_from_file()` and check for "No stats found" warnings before running the full pipeline.

## 2026-03-03 — Best-of-K selection kills sampling diversity
**What happened**: With K=100 candidates per (champion, temperature) cell, the "best by EV" selection always converged to near-chalk brackets even at high temperatures. Mean overlap was 69.7% despite temperature spread. The E8 picks were identical across 12/15 brackets.
**Root cause**: Maximizing expected score is equivalent to "find the most chalk-like bracket that still respects the temperature." With 100 samples, there's always one lucky draw where every close game went to the favorite. This undoes the diversity that sampling was supposed to provide.
**Fix**: Reduced K from 100 to 12. Still filters true garbage (random upsets that are incoherent) but doesn't have enough samples to over-optimize toward chalk.
**Rule going forward**: When using "generate many, select best" in a diversity-sensitive context, keep K low (10-20). The selection step is an adversary to diversity — it must be balanced against the need for quality filtering.

## 2026-03-03 — Model trails seed baseline on raw game-by-game accuracy
**What happened**: Game-level LOSO backtest showed model accuracy (69.9%) is slightly below seed-only baseline (72.6%). Model only beats seed in 4 of 16 seasons. Net: -29 games across 1,062.
**Root cause**: Seed is an extremely efficient predictor for lopsided matchups (1v16, 2v15). The model's value is not in binary picks but in probability calibration — knowing that a 5v12 game is 62/38 rather than just "pick the 5 seed." The bracket optimizer needs calibrated probabilities, not just binary picks.
**Implication**: The model IS adding value through probability calibration (log-loss 0.5618 on calibrated ensemble), but we should not expect it to beat seed on raw accuracy. The real test is bracket scoring against actual tournament outcomes, not game-by-game accuracy.
**Rule going forward**: Evaluate model quality by log-loss and bracket-level scoring, not game-by-game accuracy. Raw accuracy is dominated by easy matchups where seed and model agree. The model's edge is in close games and probability gradation.

## 2026-03-03 — MatchupCache was silently using the worst model
**What happened**: MatchupCache in simulate.py used `win_prob_a_logistic` with a comment saying "use best single model." But logistic had the worst log-loss of the three models. Every bracket was generated using degraded probabilities.
**Root cause**: The comment was written when the code was first created and never updated when model evaluation results came in. No validation step checked which model the cache was actually using.
**Fix**: Switched to `win_prob_a_ensemble` with optimized weights. Also added calibration.
**Rule going forward**: When a component depends on "the best model," verify which model that actually is after every retraining. Don't trust stale comments. The MatchupCache model choice should be a config parameter, not a hardcoded string.

## 2026-03-16 — First Four TBD entries resolved by string split, not model
**What happened**: `build_region_round1()` resolved `TBD_TeamA_TeamB` entries by splitting on `_` and taking the first team — an arbitrary choice unrelated to team quality.
**Root cause**: Quick placeholder logic that was never replaced with proper resolution.
**Fix**: Added `resolve_first_four()` that looks up both teams in the probability cache and picks the model favorite.
**Rule going forward**: Any time bracket entries involve play-in/First Four games, resolve them using model probabilities, not string manipulation.

## 2026-03-16 — Set-based overlap overstates bracket similarity
**What happened**: Overlap between brackets was computed using set intersection per round, which ignores position. Two brackets could have the same teams in a round but in entirely different slots (different matchup paths) and still show 100% overlap.
**Root cause**: Using `set(round_picks).intersection()` instead of position-aware comparison.
**Fix**: Switched to `zip()`-based comparison that checks each slot individually. Also changed to compute overlap against ALL previous brackets (reporting max) instead of only the last one.
**Rule going forward**: Bracket overlap must be position-aware. Two brackets with the same Sweet 16 set but different regional paths are meaningfully different and should not be penalized.

## 2026-03-16 — Ownership JSON keys must match internal round names
**What happened**: Upset validation looked up `ownership.get("s16", {})` but JSON keys were `"sweet_16"`, `"elite_eight"`, etc. Every S16/E8 ownership lookup silently fell back to uniform (1/8 = 12.5%), masking real ownership data.
**Root cause**: `parse_ownership.py` used descriptive key names while `bracket_gen.py` used terse internal round names. No validation caught the mismatch because the fallback is silent.
**Fix**: Standardized JSON keys to internal names: `r64`, `r32`, `s16`, `e8`, `f4`, `champ`.
**Rule going forward**: Ownership JSON keys must exactly match the round name strings used in bracket_gen.py's round loop (`["r64", "r32", "s16", "e8", "f4"]`) and champion/F4 selection (`"champ"`, `"f4"`). If either side changes, the other must follow.

## 2026-03-16 — Play-in losers in reach_probs inflate value scores
**What happened**: `simulate.py` generates reach probabilities for all 68 teams including both sides of First Four matchups. When `bracket_gen.py` selected F4 value picks, eliminated play-in losers (e.g. Texas) were still in the candidate pool. Texas had 0.01% F4 ownership, giving it a 34.6x value score — the highest in its region — despite being eliminated.
**Root cause**: No filtering step between the simulation output (all teams) and the bracket generator's value selection (should only consider live teams).
**Fix**: Added early First Four resolution at the top of `generate_bracket()` that identifies eliminated teams and filters them from `reach_probs` before any champion/F4 selection.
**Rule going forward**: Any code that reads `reach_probabilities_YYYY.csv` for pick selection must filter to the 64 teams that survived First Four resolution. The simulation intentionally includes all 68 for completeness, but downstream consumers must scope to the live field.
