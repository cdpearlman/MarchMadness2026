# Architecture

## System Overview

Two-layer prediction system for NCAA March Madness tournaments:

1. **Game Probability Model** — predicts win probability for any head-to-head matchup
2. **Bracket Optimizer** — uses those probabilities to fill out full tournament brackets optimized for pool scoring

## Data Pipeline

```
Barttorvik season CSVs (data/barttorvik/YYYY.csv, 40 cols each)
  │
  ├─ ingest_barttorvik.py ──→ team_stats.csv (combined, with adj_em derived)
  │
  ├─ team_matching.py ──→ team_name_mapping.csv (Barttorvik name → TeamID)
  │
  ├─ data_prep.py ──→ training_data.csv (joined games + stats)
  │
  ├─ feature_engineering.py ──→ stat differentials (TeamA − TeamB)
  │
  └─ models.py ──→ trained models + LOSO CV results + SHAP importance
        │
        └─ predict.py ──→ win probabilities for any matchup
              │
              └─ bracket_engine.py ──→ optimized bracket picks
```

### Data Source

Barttorvik (`barttorvik.com`) via manual browser CSV download. Per-season CSVs from `team-tables_each.php` endpoint with pre-tournament date filtering (end = Selection Sunday). Combined into `team_stats.csv` (6,689 rows, 42 cols, 2008-2026) via `ingest_barttorvik.py`. Seeds from Kaggle historical data stored in `data/seed_reference.csv`, merged during ingestion.

Previous: KenPom data via Kaggle (backed up as `team_stats_kenpom_backup.csv`).

## Layer 1: Game Probability Model

### Models
- **Random Forest** — best performer (0.564 log-loss, 70.3% accuracy, 0.775 AUC)
- **XGBoost** — 0.574 log-loss, 70.7% accuracy
- **Logistic Regression** — 0.575 log-loss, 69.8% accuracy
- **Ensemble** (weighted blend of all three)

### Features (24 differentials)
All features computed as `TeamA_value - TeamB_value`. Defined in `config.py` using native Barttorvik column names:
- Adjusted: adj_o, adj_d, adj_t, adj_em (derived = adj_o - adj_d)
- Four Factors offense: efg, tov_rate, oreb_rate, ftr
- Four Factors defense: def_efg, def_tov_rate, dreb_rate, def_ftr
- Shooting: two_pt_pct, three_pt_pct, ft_pct, three_fg_rate
- Opponent shooting: def_two_pt_pct, def_three_pt_pct, def_ft_pct, def_three_fg_rate
- Defense: block_rate, block_rate_allowed
- Miscellaneous: barthag
- Physical (2008+): eff_height, experience
- Meta: SeedNum

### Key Design Choices
- **Pre-tournament stats only** — Barttorvik CSVs filtered to end on Selection Sunday each year to avoid data leakage
- **Two rows per game** — each game creates two training rows (swap A/B) to prevent positional bias
- **LOSO CV** — Leave-One-Season-Out cross-validation simulates real prediction conditions
- **Missing physical features** — pre-2008 rows fill height/experience with 0 (moot now, since training starts at 2008)

## Layer 2: Bracket Optimizer

### Current Approach: Analytical Dynamic Programming (`bracket_engine.py`)
- Builds a binary tree of the full tournament bracket
- Computes path probabilities for every team reaching every round
- Uses DP to find optimal bracket picks that maximize expected pool score
- Generates diverse brackets via diversity weighting to avoid near-identical outputs
- NCAA.com scoring: R64=1pt, R32=2pts, S16=4pts, E8=8pts, F4=16pts, Championship=32pts

### Previous Approach: Monte Carlo (deprecated direction)
Monte Carlo simulation was too greedy — a 51% favorite always won over a 49% underdog, ignoring strategic value of upsets (better next-round matchups, bracket diversity). The analytical DP approach handles this by evaluating full paths through the bracket.

### Bracket Diversity
The goal is to maximize the probability that ONE bracket scores extremely well, not to have all brackets be decent. This requires:
- Diverse champion/Final Four picks across the bracket set
- Strategic upset selections where the risk/reward math favors it
- Diversity weighting parameter to control overlap between brackets

**Known limitation**: Current diversity operates only at pod level (S16 winners). The greedy EV pass from S16 upward collapses late-round picks, causing most brackets to share the same E8/F4/champion. R64 picks are never diversified. Needs structural improvement — see decisions.md for candidate approaches.

## Configuration

All paths, features, hyperparameters, and overrides live in `src/config.py`. This is the single source of truth for pipeline configuration.

## File Roles

| File | Role |
|------|------|
| `src/config.py` | Paths, features, hyperparams, manual overrides |
| `download_barttorvik.py` | Opens browser for each season's CSV download |
| `ingest_barttorvik.py` | Combines season CSVs → team_stats.csv |
| `src/team_matching.py` | Barttorvik team name → Kaggle TeamID mapping |
| `src/data_prep.py` | Joins tournament results with team stats |
| `src/feature_engineering.py` | Computes stat differentials |
| `src/models.py` | Model training, LOSO CV, SHAP analysis |
| `src/predict.py` | CLI for matchup/bracket predictions |
| `src/bracket_engine.py` | Analytical DP bracket optimization |
| `src/simulate.py` | Bracket simulation and strategy generation |
| `src/run_eval.py` | Helper to run full eval and save results |
| `tune_diversity.py` | Parameter tuning for bracket diversity weights |
