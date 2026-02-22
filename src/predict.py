"""
Bracket prediction: generate win probabilities for tournament matchups.

Usage:
  python src/predict.py                  # Predict all possible matchups for latest season
  python src/predict.py --season 2025    # Predict for specific season
  python src/predict.py --matchup "Duke" "Houston"  # Single matchup
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data_prep import load_team_stats, build_training_data
from feature_engineering import prepare_features, compute_differentials, handle_missing_features, get_feature_names
from models import train_final_models, ensemble_predict


MODEL_DIR = config.PROJECT_ROOT / "models"


def get_tournament_teams(stats: pd.DataFrame, season: int) -> pd.DataFrame:
    """Get all teams with a seed in the given season (tournament participants)."""
    mask = (stats[config.STATS_SEASON_COL] == season) & stats["SeedNum"].notna()
    teams = stats[mask].copy()
    teams = teams.sort_values("SeedNum")
    print(f"  Found {len(teams)} tournament teams for {season}")
    return teams


def predict_matchup(
    team_a: pd.Series,
    team_b: pd.Series,
    models: dict,
    scaler,
    feature_cols: list[str],
) -> dict:
    """
    Predict the outcome of a single matchup.

    Returns dict with:
      - team_a_name, team_b_name
      - team_a_seed, team_b_seed
      - win_prob_a: probability Team A wins
      - individual model probabilities
    """
    # Build a single-row DataFrame with A_ and B_ columns
    row = {}
    for col in feature_cols:
        if col == "SeedNum":
            row[f"A_{col}"] = team_a.get("SeedNum", np.nan)
            row[f"B_{col}"] = team_b.get("SeedNum", np.nan)
        else:
            row[f"A_{col}"] = team_a.get(col, np.nan)
            row[f"B_{col}"] = team_b.get(col, np.nan)

    row_df = pd.DataFrame([row])

    # Same pipeline as training: differentials → handle missing
    diff_df = pd.DataFrame()
    for col in feature_cols:
        a_col = f"A_{col}"
        b_col = f"B_{col}"
        if a_col in row_df.columns and b_col in row_df.columns:
            diff_df[f"diff_{col}"] = (
                row_df[a_col].astype(float) - row_df[b_col].astype(float)
            )

    diff_df = diff_df.fillna(0.0)

    # Ensure column order matches training
    X_pred = pd.DataFrame(columns=scaler.feature_names_in_)
    for col in X_pred.columns:
        X_pred[col] = diff_df[col] if col in diff_df.columns else 0.0

    # Get predictions from all models
    X_scaled = pd.DataFrame(
        scaler.transform(X_pred), columns=X_pred.columns
    )

    probs = {}
    for name, model in models.items():
        probs[name] = float(model.predict_proba(X_scaled)[:, 1][0])

    # Ensemble
    weights = config.ENSEMBLE_WEIGHTS
    w_sum = sum(weights)
    ens_prob = sum(
        w / w_sum * probs.get(n, 0.5)
        for n, w in zip(["logistic", "xgboost", "rf"], weights)
    )

    team_a_name = team_a.get(config.STATS_TEAM_NAME_COL, "Team A")
    team_b_name = team_b.get(config.STATS_TEAM_NAME_COL, "Team B")

    return {
        "team_a": team_a_name,
        "team_b": team_b_name,
        "seed_a": int(team_a.get("SeedNum", 0)),
        "seed_b": int(team_b.get("SeedNum", 0)),
        "win_prob_a_logistic": probs.get("logistic", 0.5),
        "win_prob_a_xgboost": probs.get("xgboost", 0.5),
        "win_prob_a_rf": probs.get("rf", 0.5),
        "win_prob_a_ensemble": ens_prob,
    }


def predict_all_matchups(
    season: int,
    stats: pd.DataFrame,
    models: dict,
    scaler,
) -> pd.DataFrame:
    """Generate win probabilities for all possible first-round matchups."""
    teams = get_tournament_teams(stats, season)
    feature_cols = config.FEATURES + ["SeedNum"]

    if len(teams) == 0:
        print(f"  ⚠️  No tournament teams found for {season}")
        return pd.DataFrame()

    # Standard bracket first-round matchups: 1v16, 2v15, ..., 8v9
    seed_matchups = [(1, 16), (2, 15), (3, 14), (4, 13),
                     (5, 12), (6, 11), (7, 10), (8, 9)]

    results = []
    for seed_a, seed_b in seed_matchups:
        a_teams = teams[teams["SeedNum"] == seed_a]
        b_teams = teams[teams["SeedNum"] == seed_b]

        for _, team_a in a_teams.iterrows():
            for _, team_b in b_teams.iterrows():
                result = predict_matchup(team_a, team_b, models, scaler, feature_cols)
                results.append(result)

    return pd.DataFrame(results)


def find_team(stats: pd.DataFrame, name: str, season: int) -> pd.Series | None:
    """Find a team by partial name match in a specific season."""
    mask = (
        stats[config.STATS_SEASON_COL] == season
    ) & (
        stats[config.STATS_TEAM_NAME_COL].str.contains(name, case=False, na=False)
    )
    matches = stats[mask]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


# ───────────────────────────────────────────────────────────────
# Save / load trained models
# ───────────────────────────────────────────────────────────────

def save_models(models: dict, scaler, path: Path | None = None) -> None:
    """Save trained models and scaler to disk."""
    if path is None:
        path = MODEL_DIR
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "trained_models.pkl", "wb") as f:
        pickle.dump({"models": models, "scaler": scaler}, f)
    print(f"  Saved models to {path / 'trained_models.pkl'}")


def load_models(path: Path | None = None) -> tuple[dict, object]:
    """Load trained models and scaler from disk."""
    if path is None:
        path = MODEL_DIR
    with open(path / "trained_models.pkl", "rb") as f:
        data = pickle.load(f)
    return data["models"], data["scaler"]


# ───────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="March Madness Bracket Predictor")
    parser.add_argument("--season", type=int, default=None,
                        help="Season to predict (default: latest)")
    parser.add_argument("--matchup", nargs=2, metavar=("TEAM_A", "TEAM_B"),
                        help="Predict a single matchup")
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-training models")
    args = parser.parse_args()

    # Load or train models
    model_file = MODEL_DIR / "trained_models.pkl"
    if model_file.exists() and not args.retrain:
        print("Loading trained models...")
        models, scaler = load_models()
    else:
        print("Training models on all historical data...")
        training_df = build_training_data()
        X, y, seasons = prepare_features(training_df)
        models, scaler = train_final_models(X, y)
        save_models(models, scaler)

    stats = load_team_stats()

    # Determine season
    season = args.season or int(stats[config.STATS_SEASON_COL].max())
    print(f"\nPredicting for season: {season}")

    if args.matchup:
        # Single matchup
        team_a = find_team(stats, args.matchup[0], season)
        team_b = find_team(stats, args.matchup[1], season)

        if team_a is None:
            print(f"  ❌ Could not find team matching '{args.matchup[0]}' in {season}")
            return
        if team_b is None:
            print(f"  ❌ Could not find team matching '{args.matchup[1]}' in {season}")
            return

        feature_cols = config.FEATURES + ["SeedNum"]
        result = predict_matchup(team_a, team_b, models, scaler, feature_cols)

        team_a_name = result["team_a"]
        team_b_name = result["team_b"]
        print(f"\n  {'='*50}")
        print(f"  {team_a_name} (#{result['seed_a']}) vs {team_b_name} (#{result['seed_b']})")
        print(f"  {'='*50}")
        print(f"  Logistic:  {result['win_prob_a_logistic']:.1%} — {1-result['win_prob_a_logistic']:.1%}")
        print(f"  XGBoost:   {result['win_prob_a_xgboost']:.1%} — {1-result['win_prob_a_xgboost']:.1%}")
        print(f"  RF:        {result['win_prob_a_rf']:.1%} — {1-result['win_prob_a_rf']:.1%}")
        print(f"  Ensemble:  {result['win_prob_a_ensemble']:.1%} — {1-result['win_prob_a_ensemble']:.1%}")

    else:
        # All first-round matchups
        results = predict_all_matchups(season, stats, models, scaler)

        if len(results) == 0:
            return

        # Print results
        print(f"\n  {'='*70}")
        print(f"  First-Round Matchup Predictions ({season})")
        print(f"  {'='*70}")

        for _, row in results.iterrows():
            winner = row["team_a"] if row["win_prob_a_ensemble"] > 0.5 else row["team_b"]
            prob = max(row["win_prob_a_ensemble"], 1 - row["win_prob_a_ensemble"])
            upset = "⚡ UPSET" if row["seed_a"] > row["seed_b"] and row["win_prob_a_ensemble"] > 0.5 else ""
            upset = upset or ("⚡ UPSET" if row["seed_b"] > row["seed_a"] and row["win_prob_a_ensemble"] < 0.5 else "")

            print(
                f"  #{row['seed_a']:2d} {row['team_a']:25s} vs #{row['seed_b']:2d} {row['team_b']:25s} "
                f"→ {winner:25s} ({prob:.1%}) {upset}"
            )

        # Save predictions
        output_path = config.PROCESSED_DIR / f"predictions_{season}.csv"
        results.to_csv(output_path, index=False)
        print(f"\n  Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
