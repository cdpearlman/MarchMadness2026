# Conventions

## Code Organization

- All source code lives in `src/`, runnable as scripts (`python src/models.py`)
- Configuration is centralized in `src/config.py` — paths, features, hyperparams, overrides
- Generated outputs go to `data/processed/`
- Trained models go to `models/`
- Root-level CSVs are raw/external data (KenPom, Kaggle tournament data)

## Validation Approach

There are no unit tests. Validation is done by:
1. Running the full pipeline end-to-end
2. Inspecting output CSVs (CV results, predictions, SHAP importance)
3. Checking SHAP values to ensure the model learns meaningful patterns, not just seed

### What to Watch For
- **Seed dominance in SHAP** — if Seed differential is overwhelmingly the top feature, the model is just predicting higher-seed wins. Other efficiency/shooting metrics should carry meaningful weight.
- **Log-loss vs accuracy** — accuracy alone can be misleading (always picking favorites gets ~70%). Log-loss and AUC-ROC are better signals.
- **Bracket similarity** — if generated brackets are >90% identical, the diversity mechanism needs tuning.

## Pipeline Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Team name matching (regenerates team_name_mapping.csv)
python src/team_matching.py

# Full model evaluation (LOSO CV + SHAP)
python src/models.py

# Predict a single matchup
python src/predict.py --matchup "Duke" "Houston"

# Generate bracket predictions for a season
python src/predict.py --season 2025

# Run bracket simulation/optimization
python src/simulate.py --season 2025 --n-sims 10000 --n-brackets 5

# Tune diversity parameter
python tune_diversity.py
```

## Data Integrity Rules

- **Never use in-tournament stats for training** — only pre-tournament metrics
- **Two rows per game** — always generate both A-vs-B and B-vs-A to avoid positional bias
- **LOSO CV only** — never evaluate on seasons included in training
- **2020 is excluded** — no tournament was played
- **Physical features (height, experience)** — only available from 2007+; fill with 0 for earlier seasons
