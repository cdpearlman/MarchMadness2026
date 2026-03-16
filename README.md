# March Madness 2026 — Predictive Model

A Python machine learning pipeline that predicts NCAA March Madness tournament game outcomes using Barttorvik efficiency metrics. The model uses **stat differentials** (Team A − Team B) as features and evaluates multiple classifiers via Leave-One-Season-Out cross-validation.

## Current Results

| Model | Log-Loss | Accuracy | AUC-ROC |
|---|---|---|---|
| **Ensemble (calibrated)** | **0.5515** | **70.0%** | **0.778** |
| Ensemble (raw) | 0.5608 | 70.0% | 0.778 |
| Random Forest | 0.5644 | 70.3% | 0.775 |
| Logistic Regression | 0.5650 | 69.8% | 0.769 |
| XGBoost | 0.6063 | 70.7% | 0.769 |

Ensemble weights optimized via LOSO CV: [0.514 LogReg, 0.0 XGB, 0.486 RF]. Probability calibration via isotonic regression. Evaluated across 17 tournament seasons (2008–2025, excl. 2020).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download Barttorvik data (opens browser for each season)
python download_barttorvik.py

# Ingest season CSVs into team_stats.csv
python ingest_barttorvik.py

# Run team name matching (generates data/processed/team_name_mapping.csv)
python src/team_matching.py

# Run full model evaluation (LOSO CV + SHAP)
python src/models.py

# Predict a single matchup
python src/predict.py --matchup "Duke" "Houston"

# Generate all first-round bracket predictions for 2026
python src/predict.py --season 2026

# Force re-train models before predicting
python src/predict.py --retrain --season 2026

# Run game-level backtesting (model vs seed baseline across all seasons)
python src/backtest.py

# Run Monte Carlo simulation to estimate per-team reach probabilities
python src/simulate.py --n-sims 50000

# Generate specific bracket strategies (Model Chalk, Value Champion, Contrarian)
python src/bracket_gen.py --ownership data/ownership_2026.json --types chalk value contrarian
```

## Project Structure

```
MarchMadness26/
├── team_stats.csv                      # Barttorvik data (6,689 rows, 42 cols, 2008-2026)
├── download_barttorvik.py              # Opens browser to download per-season CSVs
├── ingest_barttorvik.py                # Combines season CSVs → team_stats.csv
├── MNCAATourneyDetailedResults.csv     # Tournament game results (2003-2024)
├── MTeams.csv                          # TeamID → TeamName (380 teams)
├── MTeamSpellings.csv                  # Spelling variants → TeamID
├── requirements.txt
├── README.md
│
├── data/
│   ├── barttorvik/                     # Per-season Barttorvik CSVs (YYYY.csv)
│   ├── seed_reference.csv              # Historical tournament seeds (from Kaggle)
│   ├── bracket_2026.json               # Tournament bracket structure and matchups
│   └── ownership_2026.json             # Public pick percentages for EV calculation
│
├── src/
│   ├── config.py                       # All paths, features, hyperparams, overrides
│   ├── team_matching.py                # Barttorvik name → TeamID mapping
│   ├── data_prep.py                    # Joins games + stats → training rows
│   ├── feature_engineering.py          # Computes stat differentials
│   ├── models.py                       # LogReg, XGBoost, RF, Ensemble + LOSO CV + SHAP + calibration
│   ├── predict.py                      # CLI for matchup/bracket predictions (calibrated)
│   ├── backtest.py                     # Game-level LOSO backtesting vs seed baseline
│   ├── simulate.py                     # Monte Carlo tournament simulator
│   ├── bracket_gen.py                  # Bracket strategy (Chalk, Value, Contrarian)
│   └── run_eval.py                     # Helper to run full eval and save results
│
├── data/processed/
│   ├── team_name_mapping.csv           # Generated: Barttorvik name → TeamID lookup
│   ├── training_data.csv               # Generated: joined game+stats dataset
│   ├── logistic_cv_results.csv         # Per-season LOSO results
│   ├── xgboost_cv_results.csv
│   ├── rf_cv_results.csv
│   ├── ensemble_cv_results.csv
│   ├── shap_importance.csv             # Feature importance rankings
│   ├── backtest_results.csv            # Per-season model vs seed accuracy/log-loss
│   ├── reach_probabilities_YYYY.csv    # Per-team reach probabilities by round
│   ├── brackets_YYYY.json             # Generated bracket set with diversity report
│   └── predictions_2026.csv            # 2026 bracket predictions
│
├── models/
│   └── trained_models.pkl              # Pickled models + scaler + calibrators
```

## Data Pipeline

```
data/barttorvik/YYYY.csv ──→ ingest_barttorvik.py ──→ team_stats.csv
                                                           │
team_stats.csv ──────────┐                                 │
                         ├──→ team_matching.py ──→ team_name_mapping.csv
MTeamSpellings.csv ──────┘         │
                                   ▼
MNCAATourneyDetailedResults.csv → data_prep.py → training_data.csv
                                                      │
                                                      ▼
                                          feature_engineering.py
                                          (stat differentials)
                                                      │
                                                      ▼
                                               models.py
                                          (LOSO CV + training)
                                                      │
                                                      ▼
                                               predict.py
                                        (bracket predictions)
                                                       │
                                                       ▼
                                              simulate.py
                                        (Monte Carlo probabilities)
                                                       │
                                                       ▼
                                            bracket_gen.py
                                     (ownership-adjusted bracket picks)
```

## Features Used (24 differential features)

All features are computed as `TeamA_value - TeamB_value`:

| Category | Features |
|---|---|
| Adjusted Efficiency | adj_o, adj_d, adj_t, adj_em |
| Four Factors (Offense) | efg, tov_rate, oreb_rate, ftr |
| Four Factors (Defense) | def_efg, def_tov_rate, dreb_rate, def_ftr |
| Shooting | two_pt_pct, three_pt_pct, ft_pct, three_fg_rate |
| Opponent Shooting | def_two_pt_pct, def_three_pt_pct, def_ft_pct, def_three_fg_rate |
| Defense | block_rate, block_rate_allowed |
| Miscellaneous | barthag |
| Physical (2008+) | eff_height, experience |
| Meta | SeedNum |

## Key Design Decisions

- **Pre-Tournament stats only**: Barttorvik CSVs filtered to end on Selection Sunday to avoid data leakage
- **Two rows per game**: Each game creates two training rows (swap Team A/B) to prevent positional bias
- **LOSO CV**: Leave-One-Season-Out ensures no future data leaks into training
- **Champion x Temperature bracket generation**: Forces different champions across brackets, samples upset combinations at varying temperatures, stratified selection for portfolio coverage

## Data Sources

- **Team Stats**: [Barttorvik](https://barttorvik.com) — T-Rank team efficiency ratings
- **Tournament Game Results**: [Kaggle — March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data)
- **Seeds**: Extracted from Kaggle KenPom dataset (historical reference)
