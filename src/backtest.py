"""
Backtesting framework for the March Madness prediction pipeline.

Two modes:
  1. Game-level LOSO analysis: For each historical season, compute how the
     model's game-by-game predictions compare to seed-only baseline.
  2. Bracket-level backtest: For seasons with bracket JSON files and actual
     results, generate brackets and score against the real outcome.

Usage:
  python src/backtest.py                    # Game-level analysis, all seasons
  python src/backtest.py --season 2024      # Single season
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data_prep import load_team_stats, build_training_data
from feature_engineering import prepare_features
from models import loso_collect_oof_predictions, _create_model, fit_calibrators
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

ROUND_POINTS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}


def run_game_level_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    training_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run LOSO CV and compare model predictions to seed-only baseline.

    For each game, checks:
      - Would the model pick the winner? (model accuracy)
      - Would picking the lower seed work? (seed baseline)
      - Model confidence (predicted probability)
    """
    unique_seasons = sorted(seasons.unique())

    y_parts = []
    model_parts = []
    seed_parts = []
    season_parts = []

    for test_season in unique_seasons:
        train_mask = seasons != test_season
        test_mask = seasons == test_season

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index,
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )

        # Train logistic + RF, blend with optimized weights
        models = {}
        for name in ["logistic", "rf"]:
            m = _create_model(name)
            m.fit(X_train_s, y_train)
            models[name] = m

        weights = np.array(config.ENSEMBLE_WEIGHTS[:3])
        weights = weights / weights.sum()

        probas = {}
        for name in ["logistic", "xgboost", "rf"]:
            if name in models:
                probas[name] = models[name].predict_proba(X_test_s)[:, 1]
            else:
                probas[name] = np.full(len(y_test), 0.5)

        ensemble = sum(w * probas[n] for n, w in zip(["logistic", "xgboost", "rf"], weights))

        # Seed baseline: diff_SeedNum < 0 means Team A has lower seed number (= favored)
        seed_diff = X_test["diff_SeedNum"].values
        seed_pred = (seed_diff < 0).astype(float)
        seed_pred[seed_diff == 0] = 0.5

        # Only keep the "A wins" rows (Label=1 means A won) to avoid double-counting
        label_1_mask = y_test.values == 1
        y_parts.append(y_test.values[label_1_mask])
        model_parts.append(ensemble[label_1_mask])
        seed_parts.append(seed_pred[label_1_mask])
        season_parts.append(np.full(label_1_mask.sum(), test_season))

    y_all = np.concatenate(y_parts)
    model_all = np.concatenate(model_parts)
    seed_all = np.concatenate(seed_parts)
    season_all = np.concatenate(season_parts)

    results = []
    for s in sorted(set(season_all)):
        mask = season_all == s
        y_s = y_all[mask]
        m_s = model_all[mask]
        seed_s = seed_all[mask]
        n_games = len(y_s)

        model_correct = (m_s >= 0.5).sum()
        seed_correct = (seed_s >= 0.5).sum()
        model_acc = model_correct / n_games
        seed_acc = seed_correct / n_games

        model_ll = log_loss(y_s, np.clip(m_s, 1e-15, 1 - 1e-15), labels=[0, 1])

        results.append({
            "season": int(s),
            "n_games": n_games,
            "model_correct": int(model_correct),
            "model_accuracy": round(model_acc, 4),
            "seed_correct": int(seed_correct),
            "seed_accuracy": round(seed_acc, 4),
            "model_log_loss": round(model_ll, 4),
            "model_advantage": int(model_correct - seed_correct),
        })

    return pd.DataFrame(results)


def print_game_level_report(df: pd.DataFrame) -> None:
    """Pretty-print the game-level backtest results."""
    print(f"\n{'='*80}")
    print(f"  GAME-LEVEL BACKTEST: Model vs Seed Baseline (LOSO)")
    print(f"{'='*80}")
    print(f"  {'Season':<8} {'Games':>5} {'Model':>8} {'Seed':>8} {'Adv':>5} {'Model LL':>10}")
    print(f"  {'-'*50}")

    for _, row in df.iterrows():
        adv_str = f"+{row['model_advantage']}" if row['model_advantage'] > 0 else str(row['model_advantage'])
        print(
            f"  {int(row['season']):<8} {int(row['n_games']):>5} "
            f"{row['model_accuracy']:>7.1%} {row['seed_accuracy']:>7.1%} "
            f"{adv_str:>5} {row['model_log_loss']:>10.4f}"
        )

    print(f"  {'-'*50}")
    total_games = df['n_games'].sum()
    total_model = df['model_correct'].sum()
    total_seed = df['seed_correct'].sum()
    total_adv = total_model - total_seed
    avg_model_acc = total_model / total_games
    avg_seed_acc = total_seed / total_games
    avg_ll = df['model_log_loss'].mean()

    print(
        f"  {'TOTAL':<8} {int(total_games):>5} "
        f"{avg_model_acc:>7.1%} {avg_seed_acc:>7.1%} "
        f"{'+' + str(total_adv) if total_adv > 0 else str(total_adv):>5} {avg_ll:>10.4f}"
    )

    model_wins = (df['model_advantage'] > 0).sum()
    ties = (df['model_advantage'] == 0).sum()
    seed_wins = (df['model_advantage'] < 0).sum()
    print(f"\n  Model beats seed in {model_wins}/{len(df)} seasons "
          f"(ties: {ties}, seed wins: {seed_wins})")
    print(f"  Net advantage: {total_adv} more correct picks across {int(total_games)} games")


def main():
    parser = argparse.ArgumentParser(description="March Madness Backtester")
    parser.add_argument("--season", type=int, default=None,
                        help="Single season to backtest (default: all)")
    args = parser.parse_args()

    print("Step 1: Building training data...")
    training_df = build_training_data()

    print("\nStep 2: Computing features...")
    X, y, seasons = prepare_features(training_df)

    if args.season:
        mask = seasons == args.season
        if mask.sum() == 0:
            print(f"  [!] No data for season {args.season}")
            return
        print(f"\nRunning backtest for season {args.season} only...")
    else:
        print(f"\nRunning game-level backtest across all LOSO seasons...")

    results = run_game_level_backtest(X, y, seasons, training_df)

    if args.season:
        results = results[results['season'] == args.season]

    print_game_level_report(results)

    output_path = config.PROCESSED_DIR / "backtest_results.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n  Saved results -> {output_path}")


if __name__ == "__main__":
    main()
