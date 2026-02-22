# March Madness 2026 — Predictive Model

A Python machine learning pipeline that predicts NCAA March Madness tournament game outcomes using KenPom efficiency metrics. The model uses **stat differentials** (Team A − Team B) as features and evaluates multiple classifiers via Leave-One-Season-Out cross-validation.

## Current Results

| Model | Log-Loss | Accuracy | AUC-ROC |
|---|---|---|---|
| **Logistic Regression** | **0.364** | **83.3%** | **0.920** |
| XGBoost | 0.399 | 83.1% | 0.899 |
| Ensemble (LR+XGB+RF) | 0.394 | 83.3% | 0.908 |

Evaluated via LOSO CV across 21 tournament seasons (2003–2024, excl. 2020).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run team name matching (generates data/processed/team_name_mapping.csv)
python src/team_matching.py

# Run full model evaluation (LOSO CV + SHAP)
python src/models.py

# Predict a single matchup
python src/predict.py --matchup "Duke" "Houston"

# Generate all first-round bracket predictions for 2025
python src/predict.py --season 2025

# Force re-train models before predicting
python src/predict.py --retrain --season 2025
```

## Project Structure

```
MarchMadness26/
├── team_stats.csv                      # KenPom data (8,315 rows, 165 cols, 2002-2025)
├── MNCAATourneyDetailedResults.csv     # Tournament game results (1,382 games, 2003-2024)
├── MTeams.csv                          # TeamID → TeamName (380 teams)
├── MTeamSpellings.csv                  # Spelling variants → TeamID (1,177 entries)
├── requirements.txt
├── README.md
├── HANDOFF.md                          # Agent continuation guide
│
├── src/
│   ├── config.py                       # All paths, features, hyperparams, overrides
│   ├── team_matching.py                # ESPN name → TeamID mapping
│   ├── data_prep.py                    # Joins games + stats → training rows
│   ├── feature_engineering.py          # Computes stat differentials
│   ├── models.py                       # LogReg, XGBoost, RF, Ensemble + LOSO CV + SHAP
│   ├── predict.py                      # CLI for matchup/bracket predictions
│   └── run_eval.py                     # Helper to run full eval and save results
│
├── data/processed/
│   ├── team_name_mapping.csv           # Generated: ESPN → TeamID lookup
│   ├── training_data.csv               # Generated: joined game+stats dataset
│   ├── logistic_cv_results.csv         # Per-season LOSO results
│   ├── xgboost_cv_results.csv
│   ├── rf_cv_results.csv
│   ├── ensemble_cv_results.csv
│   ├── shap_importance.csv             # Feature importance rankings
│   └── predictions_2025.csv            # 2025 bracket predictions
│
└── models/
    └── trained_models.pkl              # Pickled final models + scaler
```

## Data Pipeline

```
team_stats.csv ──────────┐
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
```

## Features Used (24 differential features)

All features are computed as `TeamA_value - TeamB_value`:

| Category | Features |
|---|---|
| Efficiency (Pre-Tournament) | AdjOE, AdjDE, AdjTempo, AdjEM |
| Four Factors (Offense) | eFGPct, TOPct, ORPct, FTRate |
| Shooting | FG2Pct, FG3Pct, FTPct, FG3Rate |
| Opponent Shooting | OppFG2Pct, OppFG3Pct, OppFTPct, OppFG3Rate |
| Steal/Block | StlRate, OppStlRate, BlockPct, OppBlockPct |
| Miscellaneous | Net Rating |
| Physical (2007+) | EffectiveHeight, Experience |
| Meta | Seed (numeric) |

## Key Design Decisions

- **Pre-Tournament stats only**: Uses `Pre-Tournament.AdjOE` etc. to avoid data leakage from tournament results
- **Two rows per game**: Each game creates two training rows (swap Team A/B) to prevent positional bias
- **LOSO CV**: Leave-One-Season-Out ensures no future data leaks into training, simulates real prediction conditions
- **Missing physical features**: Pre-2007 rows fill height/experience differentials with 0 (no advantage assumed)

## Data Sources

- **KenPom Stats**: [Kaggle — Jonathan Pilafas](https://www.kaggle.com/datasets/jonathanpilafas/2024-march-madness-statistical-analysis)
- **Tournament Game Results**: [Kaggle — March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data)
