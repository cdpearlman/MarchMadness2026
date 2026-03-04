# Session Log

<!-- Append-only. Add a new entry after each substantive work session. -->

## 2026-03-03 — Bootstrap
**Area**: Project setup
**Work done**: Ran ContextKit bootstrap interview, generated memory system
**Decisions made**: Use Cursor routing file (`.cursor/rules/contextkit.mdc`); documented current architecture (analytical DP over Monte Carlo); captured critical pitfalls (seed bias, bracket diversity)
**Memory created**: architecture.md, conventions.md, sessions.md, decisions.md, lessons.md
**Open threads**: Barttorvik data source investigation; model accuracy tuning; 2025 bracket generation validation

## 2026-03-03 — Barttorvik Data Source Investigation
**Area**: Data pipeline / data sourcing
**Work done**: Deep investigation of barttorvik.com as replacement for Kaggle/KenPom data source. Researched export options, mapped all 24 features, tested programmatic access (requests, cloudscraper, Playwright — all blocked by JS verification). Found cbbdata R package source code revealing barttorvik's internal CSV URL patterns and exact column mappings. Confirmed two working browser-download CSV URLs:
- `trank.php?csv=1` (37 cols) — main stats
- `team-tables_each.php?csv=1` (40 cols) — same + height/experience data

**Decisions made**: Manual browser download is the viable access method; automated scraping is blocked. Team-tables CSV (40 cols) is the recommended source. 20/24 features map directly, 2 are derivable, only StlRate/OppStlRate are truly missing.
**Open threads**:
- Verify barttorvik cols 32/33 are EffectiveHeight/Experience (download a known season and cross-check)
- Decide: drop StlRate/OppStlRate or substitute with tov_rate/def_tov_rate
- Build ingestion script to map barttorvik CSV → team_stats.csv format
- Investigate suspicious UConn 2024 KenPom data (negative net rating for the champion — possible name matching bug)
- Historical coverage starts ~2008 (lose 2003-2007 training seasons)

## 2026-03-03 — Barttorvik Data Switch (Full Implementation)
**Area**: Data pipeline — complete data source migration
**Work done**:
- Evaluated all alternatives (KenPom API $95/yr, CFBD API, manual, hybrid). Chose Barttorvik as best free option.
- Created `download_barttorvik.py` — opens browser for each season CSV with pre-tournament date filtering (end = Selection Sunday)
- Downloaded all 19 seasons (2008-2026) of `team-tables_each.php` CSVs (40 cols each) to `data/barttorvik/`
- Identified full 40-column mapping using `cbbdata` R package source (`cbd_torvik_team_factors.R`). Cols 0-36 match trank.php, cols 37-39 are eff_height/experience/unknown.
- Created `ingest_barttorvik.py` — reads season CSVs, assigns column names, derives adj_em, writes combined `team_stats.csv` (6,689 rows)
- Updated `src/config.py` — all feature names now use native Barttorvik column names. 23 features + SeedNum + adj_em (derived). Dropped StlRate/OppStlRate (unavailable). Added defensive four factors (def_efg, def_tov_rate, dreb_rate, def_ftr) and barthag. Net gain of 3 features.
- Ran team_matching: 369/370 matched. Added 15 new MANUAL_OVERRIDES. Fixed old bug: "ul monroe" was mapped to 1349 (Rice) instead of 1419 (ULM).
- Extracted seed reference from old KenPom data → `data/seed_reference.csv`, merged into new team_stats.csv (1,960 tournament team-seasons with seeds 1-16).
- Fixed bug: `feature_engineering.py` never computed diff_SeedNum (only diffed config.FEATURES, which excluded SeedNum). Added SeedNum handling.
- Fixed emoji encoding crashes in `data_prep.py` and `team_matching.py` (Windows cp1252 can't encode unicode emojis).
- Retrained all models via LOSO CV.
**Results**: RF best — 0.564 log-loss, 70.3% accuracy, 0.775 AUC. SHAP top-5: adj_em, adj_d, ft_pct, oreb_rate, adj_o. Seed NOT in top-10 (healthy). Old KenPom baseline (0.364 LL, 83.3% acc) was likely inflated by data leakage.
**Decisions made**: Barttorvik switch confirmed and implemented. Old KenPom data backed up as `team_stats_kenpom_backup.csv`.
**Open threads**:
- Model accuracy is reasonable (~70%) but could be improved with hyperparameter tuning
- Verify barttorvik column mapping for cols 9-14 (ftr/oreb ordering vs alternative layouts)
- 2026 tournament seeds need manual entry after Selection Sunday (March 15)
- `predict.py`, `bracket_engine.py`, `simulate.py` not yet tested with new data (config-driven, should work)
- Consider adding assist_rate, def_assist_rate as features (available but not currently used)

## 2026-03-03 — First Bracket Engine Run (Post-Barttorvik Refactor)
**Area**: Bracket engine pipeline, bug fixes
**Work done**:
- Ran bracket engine for 2025 season using `data/bracket_2025.json`
- Fixed critical bug in `bracket_engine.py` `compute_all_path_probs_tree()`: the right-side team loop used a stale `p_a` variable (closure over last left-side team) instead of looking up `node.left.node_probs[t_a.name]` for each opponent. This caused all right-side-of-bracket teams to get near-zero reach probabilities (championship probs were 0.0% for ALL teams before fix)
- Added `bracket_aliases` dict in `simulate.py` to map bracket JSON names to Barttorvik names (Ole Miss->Mississippi, UConn->Connecticut, Norfolk State->Norfolk St., etc.)
- Improved partial name matching in `find_stats()` to prefer closest-length match (avoids "alabama" matching "alabama state")
- Fixed remaining emoji/Unicode in `simulate.py` (arrow chars, em-dash) and `predict.py` (warning/error emoji)
- Retrained model with `--retrain` flag since old pkl had stale KenPom feature names
**Results**: Houston 28.6%, Auburn 27.8%, Duke 18.2% championship probability. 5 brackets generated with 63% mean overlap, 34% min overlap. B1/B3 have 97% overlap (near-duplicate).
**Decisions made**: MatchupCache still uses logistic regression only (not ensemble/RF) — flagged but not changed
**Open threads**:
- **Bracket diversity convergence**: 4/5 brackets pick same E8/F4/champion (Houston). B1/B3 are 97% overlap. R64 is identical across all brackets. Pod-level diversity alone is insufficient — need diversification at E8/F4/champion level too
- MatchupCache uses `win_prob_a_logistic` but RF was best performer — consider switching to ensemble or RF
- Consider increasing `top_k_per_pod` beyond 2 for more combinatorial diversity
- Model hyperparameter tuning still pending

## 2026-03-03 — Bracket Diversity Overhaul (Champion x Temperature Stratified Sampling)
**Area**: Bracket engine (`bracket_engine.py`, `simulate.py`, `config.py`)
**Work done**:
- Replaced pod-level-only diversity system with champion x temperature stratified sampling architecture
- Added `forced_champion` parameter to `simulate_bracket_with_temperature` — forces champion through every matchup on their path, samples all other games with temperature-adjusted coin flips
- Refactored `run_analytical()` to use new pipeline: select top N champions by P(championship), then for each (champion, temp) cell generate K candidates and keep best by EV
- Added config params: `BRACKET_N_CHAMPIONS`, `BRACKET_TEMPERATURE_TIERS`, `BRACKET_CANDIDATES_PER_CELL`
- Added CLI flags: `--n-champions`, `--temperatures`, `--candidates-per-cell`
- Ran two rounds of diversity experiments (10 configs total) to tune parameters
- Old pod-based functions left in `bracket_engine.py` but no longer called
**Results**: Mean overlap dropped from 69.7% (K=100) to 56.8% (final config). 5 unique champions, 5 unique F4 configs, 6 unique E8 configs across 15 brackets. EV range 73.4-104.5.
**Final config**: `BRACKET_TEMPERATURE_TIERS = [0.4, 0.9, 1.6]`, `BRACKET_CANDIDATES_PER_CELL = 12`, `BRACKET_N_CHAMPIONS = 5`
**Decisions made**: Champion x temperature grid over pod-level diversity; sampling over deterministic EV; stratified selection over unconstrained
**Open threads**:
- MatchupCache still uses `win_prob_a_logistic` — consider switching to ensemble or RF (RF was best performer)
- Model hyperparameter tuning still pending
- E8 convergence within same-champion brackets remains — could explore forced F4 diversity as future enhancement

## 2026-03-03 — Pipeline Audit & Model Improvements
**Area**: Full pipeline evaluation, model layer, bracket engine
**Work done**:
- Comprehensive evaluation of pipeline vs state-of-the-art March Madness prediction (FiveThirtyEight, KenPom log5, Kaggle winners, academic papers on bracket pool optimization)
- **Fixed MatchupCache**: switched from `win_prob_a_logistic` to `win_prob_a_ensemble` in `simulate.py` (logistic was the worst model; ensemble is now best)
- **Optimized ensemble weights via LOSO CV**: scipy.optimize found [0.5139, 0.0, 0.4861] — XGBoost gets zero weight. Ensemble log-loss improved from 0.5723 to 0.5608, now the best model overall
- **Added probability calibration**: isotonic regression on LOSO OOF predictions, stored in pkl alongside models. Further improved ensemble log-loss from 0.5610 to 0.5515. Calibrators flow through predict_matchup -> MatchupCache -> bracket engine
- **Fixed non-deterministic RNG**: replaced `hash(champ)` with `hashlib.md5()` in bracket_engine.py for reproducible bracket generation
- **Added F4 diversity**: `run_analytical()` now forces alternative regional winners at higher temperature tiers. For each champion, temp_idx=1 forces the #2 team in one non-champion region, temp_idx=2 forces #2 in two regions
- **Built backtesting framework** (`src/backtest.py`): game-level LOSO comparison of model vs seed-only baseline
- **Regenerated SHAP importance CSV** with current Barttorvik feature names
- Removed deprecated `use_label_encoder` from XGBoost params
- Updated `load_models()`/`save_models()` to 3-tuple (models, scaler, calibrators) across predict.py, simulate.py, tune_diversity.py
**Results**:
- LOSO CV: ensemble 0.5608 LL, 70.0% acc, 0.778 AUC (best model). RF 0.5644, Logistic 0.5650, XGBoost 0.6063
- Calibration: additional 0.0095 LL improvement
- Bracket diversity: 55.5% mean overlap (down from 63%), 41.6% min overlap, F4 compositions now vary within same champion
- SHAP top-5: diff_adj_em, diff_adj_d, diff_ft_pct, diff_oreb_rate, diff_adj_o. Seed NOT in top-10
- Backtest: model accuracy 69.9% vs seed baseline 72.6% — model trails on raw picks but provides calibrated probabilities
**Decisions made**: Ensemble weights optimized; XGBoost zeroed out; calibration added; MatchupCache switched to ensemble
**Open threads**:
- **Contrarian/public-pick integration**: The biggest strategic gap. Model optimizes EV but doesn't account for what other pool participants pick. Public pick data from ESPN/CBS could add a leverage axis
- **Model vs seed accuracy gap**: Model trails seed on raw game picks (69.9% vs 72.6%). Its value is in probability calibration, not binary accuracy. Need to investigate whether additional features (SOS, preseason rankings, Vegas lines) could close this gap
- **Bracket files for backtesting**: Only have bracket_2025.json. Need bracket files for past seasons to do proper bracket-level backtest scoring
- **Hyperparameter tuning**: Still not done for any of the three models
- **Preseason rankings as a feature**: FiveThirtyEight uses AP/coaches poll as 25% of signal. Available historically, free, could help
- 2026 bracket file needed after Selection Sunday (March 15)
