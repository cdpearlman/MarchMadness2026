# Agent Handoff — March Madness 2026

This document provides full context for any agent continuing work on this project. Read this before making changes.

## What Exists

A working end-to-end March Madness prediction pipeline. All modules run, models are trained, and 2025 bracket predictions are generated. The pipeline achieves **83.3% accuracy** and **0.364 log-loss** via Leave-One-Season-Out CV (best: Logistic Regression).

### Module Map

| File | Responsibility | Key Functions |
|---|---|---|
| `src/config.py` | Central config — **start here** | Feature lists, file paths, hyperparams, manual overrides |
| `src/team_matching.py` | ESPN team name → Mania TeamID | `match_teams()`, `load_mapping()` |
| `src/data_prep.py` | Join games with team stats | `load_team_stats()`, `build_training_data()` |
| `src/feature_engineering.py` | Compute stat differentials | `prepare_features()` → returns `(X, y, seasons)` |
| `src/models.py` | Train, evaluate, SHAP | `loso_cv()`, `train_final_models()`, `compute_shap_importance()` |
| `src/predict.py` | Bracket prediction CLI | `predict_matchup()`, `predict_all_matchups()`, model save/load |
| `src/run_eval.py` | Helper script | Runs full pipeline, saves all CSV results |

### Data Flow

```
config.py defines everything
        ↓
team_matching.py → data/processed/team_name_mapping.csv
        ↓
data_prep.py → builds per-game training rows with both teams' stats
        ↓
feature_engineering.py → computes diff_* columns (TeamA - TeamB)
        ↓
models.py → trains LogReg/XGBoost/RF, runs LOSO CV, SHAP
        ↓
predict.py → generates predictions for any season/matchup
```

### Generated Artifacts

All in `data/processed/`:
- `team_name_mapping.csv` — 371 ESPN names → TeamIDs
- `training_data.csv` — 2,764 rows (1,382 games × 2)
- `{logistic,xgboost,rf,ensemble}_cv_results.csv` — per-season metrics
- `shap_importance.csv` — ranked feature importance
- `predictions_2025.csv` — all seed-pair matchup probabilities

Trained models in `models/trained_models.pkl` (pickle with dict of models + StandardScaler).

---

## Open Questions & Decisions Needed

### 1. Feature Redundancy
`diff_Net Rating` dominates SHAP importance (2.614 vs 1.070 for #2). This metric may be highly correlated with `Pre-Tournament.AdjEM` (adjusted efficiency margin). Consider:
- **Option A**: Keep both and let the model handle multicollinearity
- **Option B**: Drop `Net Rating` and see if other features become more influential
- **Option C**: Run Variance Inflation Factor (VIF) analysis to identify and prune redundant features

### 2. Ensemble Weights Are Not Optimized
Current weights in `config.py` are hardcoded `[0.30, 0.45, 0.25]` (LR, XGB, RF). Since LogReg is clearly the best single model, the ensemble underperforms it. Consider:
- Optimize weights via grid search during LOSO CV
- Or just use LogReg alone (simplest, best log-loss)
- Or use stacking (train a meta-learner on base model predictions)

### 3. Pre-Tournament vs Full-Season Stats
Currently using `Pre-Tournament.AdjOE/AdjDE/AdjTempo/AdjEM` for efficiency metrics, but full-season stats for Four Factors and shooting splits. This is inconsistent — ideally ALL features should be pre-tournament variants. Check if `team_stats.csv` has pre-tournament versions of the other features, or if they're effectively the same (most regular season stats shouldn't change much from tournament inclusion).

### 4. Hyperparameter Tuning Not Done
All model hyperparams are defaults from `config.py`. A proper Bayesian optimization or grid search over:
- XGBoost: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`
- LogReg: `C` (regularization strength)
- RF: `n_estimators`, `max_depth`, `min_samples_leaf`

This could improve performance by 1-3 percentage points.

### 5. 2025 Tournament Bracket Not Yet Available
`predict.py --season 2025` generates all possible matchups per seed pair, but the **actual 2025 bracket** (which specific teams got which seeds and which region) isn't in the data yet. When the bracket is announced, someone needs to:
- Update `team_stats.csv` with final 2025 seed assignments
- Or manually specify the bracket matchups
- Or add a bracket input mechanism to `predict.py`

### 6. The 14 Dropped Rows
`data_prep.py` reports "Dropped 14 rows with no TeamID" every run. These are teams in `team_stats.csv` with no mapping. They're likely defunct or FCS teams that never appeared in the tournament, but this should be investigated and documented.

---

## Things Still Missing

### Must-Have Before Tournament
- [ ] **Actual 2025/2026 bracket input** — `predict.py` currently does all-pairs within seed groups, not region-specific matchups
- [ ] **Calibration analysis** — Are the predicted probabilities well-calibrated? Plot reliability diagram
- [ ] **Monte Carlo bracket simulation** — Use probabilities to simulate thousands of full brackets and pick the most likely path

### Nice-to-Have Improvements
- [ ] **Hyperparameter tuning** (see Open Question #4)
- [ ] **Feature selection** — Forward/backward selection or recursive feature elimination
- [ ] **Additional models** — Neural network (MLP), LightGBM, CatBoost
- [ ] **Regular season augmentation** — Add conference tournament games to increase training data
- [ ] **Temporal weighting** — Weight recent seasons more heavily than older ones
- [ ] **Upset detection** — Flag matchups with 40-55% probabilities as potential upsets
- [ ] **Visualization** — Bracket visualization, SHAP waterfall plots, calibration curves
- [ ] **Test suite** — Automated tests for team matching, data integrity, model sanity
- [ ] **Notebook (EDA)** — Exploratory data analysis with visualizations
- [ ] **Seed-based baseline** — Implement a simple seed-probability baseline for comparison (e.g., 1-seed beats 16-seed 99% of the time historically)

### Known Technical Debt
- `run_eval.py` duplicates logic from `models.py` — could be consolidated
- `predict.py` loads data from scratch every time — could cache or use saved training data
- No `__init__.py` in `src/` — imports use `sys.path.insert` hack
- Model persistence uses `pickle` — fragile across library versions; consider `joblib` or ONNX

---

## How to Navigate

### "I want to change which features are used"
→ Edit `src/config.py`, specifically the `FEATURES_*` lists. The column names must exactly match headers in `team_stats.csv`.

### "I want to add a new model"
→ Edit `src/models.py`:
1. Add a factory function like `make_lightgbm()`
2. Register it in `_create_model()` and `_get_models()`
3. Add hyperparams to `config.py`

### "I want to add a new data source"
→ Edit `src/data_prep.py`:
1. Load the new CSV
2. Join it with the existing stats by `(TeamID, Season)`
3. Add new column names to `config.FEATURES`

### "I want to predict a different year"
→ Run `python src/predict.py --season YYYY` (requires `team_stats.csv` to have that season's data)

### "I want to update the team name overrides"
→ Edit `config.MANUAL_OVERRIDES` in `src/config.py`, then re-run `python src/team_matching.py`

### "I want to understand why a prediction was made"
→ Use SHAP: `src/models.py` has `compute_shap_importance()`. For a per-prediction explanation, add `shap.force_plot()` to `predict.py`.

---

## Environment

- **Python**: 3.10+ required (uses `dict[str, int]` type hints)
- **OS**: Developed on Windows, paths use `pathlib.Path` (cross-platform)
- **Dependencies**: See `requirements.txt` — pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap, pyyaml
- **Data**: All raw CSVs live in project root (not in a `data/raw/` subfolder)
