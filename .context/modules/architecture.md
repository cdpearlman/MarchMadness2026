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
  └─ models.py ──→ trained models + LOSO CV + SHAP + calibrators + ensemble weight optimization
        │
        └─ predict.py ──→ win probabilities for any matchup (calibrated)
              │
              └─ bracket_engine.py ──→ optimized bracket picks
```

### Data Source

Barttorvik (`barttorvik.com`) via manual browser CSV download. Per-season CSVs from `team-tables_each.php` endpoint with pre-tournament date filtering (end = Selection Sunday). Combined into `team_stats.csv` (6,689 rows, 42 cols, 2008-2026) via `ingest_barttorvik.py`. Seeds from Kaggle historical data stored in `data/seed_reference.csv`, merged during ingestion.

Previous: KenPom data via Kaggle (backed up as `team_stats_kenpom_backup.csv`).

## Layer 1: Game Probability Model

### Models
- **Ensemble** — best performer (0.5608 log-loss, 70.0% accuracy, 0.778 AUC). Optimized weights: [0.5139 LogReg, 0.0 XGB, 0.4861 RF]
- **Random Forest** — 0.5644 log-loss
- **Logistic Regression** — 0.5650 log-loss
- **XGBoost** — 0.6063 log-loss (zeroed out in ensemble)
- **Probability Calibration** — isotonic regression on LOSO OOF predictions. Reduces ensemble log-loss by ~0.01. Calibrators stored in `trained_models.pkl` alongside models and scaler. Applied transparently in `predict_matchup()`.

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

## Layer 2: Bracket Engine v1.5 — Probabilistic Portfolio Generation

### Architecture: 4-Stage Pipeline (`src/bracket_gen.py`)

```
Stage 1: Champion Pool
  Select top N teams covering ~80% of championship probability mass.
  Champions sampled proportionally (weighted random, not deterministic).

Stage 2: Bracket Generation (15,000 candidates)
  For each bracket: sample champion, draw temperature from stratified tiers.
  Each game resolved probabilistically: p_flip = upset_score^(1/temperature)
  where upset_score = p_underdog * (1 - ownership).
  Only champion's path is locked; everything else is probabilistic.

Stage 3: Tournament Simulation (50,000 Monte Carlo sims)
  simulate_bracket_raw() returns (n_sims, 63) winner matrix.
  Pure model probabilities — no ownership influence.

Stage 4: Greedy Portfolio Selection (25 brackets)
  Score all 15K candidates against 50K sims using edge-clamped leverage.
  Greedily select brackets maximizing E[max score] (submodular optimization).
```

### Edge-Clamped Leverage Scoring
Standard ESPN scoring (1/2/4/8/16/32) is modified per-pick by an edge multiplier:
- `edge = model_reach_probability / field_ownership`
- `weight = min(EDGE_CAP, max(1.0, edge))` — boost only where model > field, cap at 3.0x
- Prevents: chalk convergence (model-only E[max]) and longshot amplification (pure 1/ownership)

### Key Parameters (in `config.py`)
- `BRACKET_N_TOTAL = 15,000` — candidate brackets generated
- `BRACKET_N_SIMS = 50,000` — Monte Carlo simulations for evaluation
- `BRACKET_N_PORTFOLIO = 25` — portfolio size (ESPN max; take prefixes for Yahoo/single-entry)
- `BRACKET_CHAMP_CUMULATIVE_CUTOFF = 0.80` — champion pool covers 80% of championship mass
- `BRACKET_P_FLOOR = 0.20` — minimum upset probability to allow flipping
- `BRACKET_TEMP_TIERS` — temperature distribution: 30% chalk (0.1-0.3), 40% moderate (0.5-1.0), 30% contrarian (1.5-3.0)
- `BRACKET_EDGE_CAP = 3.0` — max leverage multiplier

### Scoring: ESPN Standard
R64=1pt, R32=2pts, S16=4pts, E8=8pts, F4=16pts, Championship=32pts

### Previous Approaches (deprecated)
- **Analytical DP** (`bracket_engine.py`, still in repo): Built binary tree, computed path probs, used DP for optimal picks. Scrapped because it optimized globally instead of reacting to ownership at individual pick level.
- **Champion x Temperature grid** (v1.0): Deterministic grid of champions x temperature tiers. Scrapped because best-of-K selection within cells converged to chalk (K too high = over-optimization).

### Backtesting

`src/backtest.py` provides game-level LOSO backtesting. For each held-out season, trains models on remaining seasons, predicts all tournament games, and compares model accuracy/log-loss against a seed-only baseline. Output saved to `data/processed/backtest_results.csv`.

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
| `src/backtest.py` | Game-level LOSO backtesting vs seed baseline |
| `tune_diversity.py` | Parameter tuning for bracket diversity weights |
