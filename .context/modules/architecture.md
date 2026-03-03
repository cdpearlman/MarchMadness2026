# Architecture

## System Overview

Two-layer prediction system for NCAA March Madness tournaments:

1. **Game Probability Model** — predicts win probability for any head-to-head matchup
2. **Bracket Optimizer** — uses those probabilities to fill out full tournament brackets optimized for pool scoring

## Data Pipeline

```
Raw CSVs (KenPom stats, tournament results, team IDs/spellings)
  │
  ├─ team_matching.py ──→ team_name_mapping.csv (ESPN name → TeamID)
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

Currently: KenPom data via Kaggle (`team_stats.csv`, 8,315 rows, 165 cols, 2002–2025).
Under consideration: switching to Barttorvik for better data freshness. This would ideally be a CSV source + column name swap without changing the pipeline logic.

## Layer 1: Game Probability Model

### Models
- **Logistic Regression** — best performer (0.364 log-loss, 83.3% accuracy, 0.920 AUC)
- **XGBoost** — 0.399 log-loss, 83.1% accuracy
- **Random Forest** — supporting model
- **Ensemble** (weighted blend of all three)

### Features (24 differentials)
All features computed as `TeamA_value - TeamB_value`. Defined in `config.py`:
- Adjusted efficiency (pre-tournament): AdjOE, AdjDE, AdjTempo, AdjEM
- Four Factors offense: eFGPct, TOPct, ORPct, FTRate
- Shooting splits: FG2Pct, FG3Pct, FTPct, FG3Rate
- Opponent shooting: OppFG2Pct, OppFG3Pct, OppFTPct, OppFG3Rate
- Steal/block: StlRate, OppStlRate, BlockPct, OppBlockPct
- Miscellaneous: Net Rating
- Physical (2007+): EffectiveHeight, Experience
- Meta: Seed (numeric)

### Key Design Choices
- **Pre-tournament stats only** — uses `Pre-Tournament.AdjOE` etc. to avoid data leakage
- **Two rows per game** — each game creates two training rows (swap A/B) to prevent positional bias
- **LOSO CV** — Leave-One-Season-Out cross-validation simulates real prediction conditions
- **Missing physical features** — pre-2007 rows fill height/experience with 0

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

## Configuration

All paths, features, hyperparameters, and overrides live in `src/config.py`. This is the single source of truth for pipeline configuration.

## File Roles

| File | Role |
|------|------|
| `src/config.py` | Paths, features, hyperparams, manual overrides |
| `src/team_matching.py` | ESPN team name → Kaggle TeamID mapping |
| `src/data_prep.py` | Joins tournament results with team stats |
| `src/feature_engineering.py` | Computes stat differentials |
| `src/models.py` | Model training, LOSO CV, SHAP analysis |
| `src/predict.py` | CLI for matchup/bracket predictions |
| `src/bracket_engine.py` | Analytical DP bracket optimization |
| `src/simulate.py` | Bracket simulation and strategy generation |
| `src/run_eval.py` | Helper to run full eval and save results |
| `tune_diversity.py` | Parameter tuning for bracket diversity weights |
