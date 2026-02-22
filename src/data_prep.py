"""
Data preparation: join tournament results with team stats.

Pipeline:
  1. Load team_name_mapping.csv (ESPN → TeamID)
  2. Load team_stats.csv with feature columns
  3. Load MNCAATourneyDetailedResults.csv 
  4. For each game, look up both teams' stats for that season
  5. Create two rows per game (swap Team A / B) to avoid positional bias
  6. Output training_data.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from team_matching import load_mapping


def load_team_stats() -> pd.DataFrame:
    """
    Load team_stats.csv, attach TeamID via the name mapping, and keep
    only the columns we need (features + identifiers).
    """
    mapping = load_mapping()
    espn_to_id = dict(zip(mapping["ESPNName"], mapping["TeamID"]))

    # Read all columns; we'll filter later
    stats = pd.read_csv(config.TEAM_STATS_CSV)
    stats["TeamID"] = (
        stats[config.STATS_TEAM_NAME_COL].map(espn_to_id).astype("Int64")
    )

    # Drop rows with no TeamID (teams not in mapping — shouldn't happen)
    before = len(stats)
    stats = stats.dropna(subset=["TeamID"])
    if len(stats) < before:
        print(f"  ⚠️  Dropped {before - len(stats)} rows with no TeamID")

    stats["TeamID"] = stats["TeamID"].astype(int)
    stats[config.STATS_SEASON_COL] = stats[config.STATS_SEASON_COL].astype(int)

    # Parse Seed: extract numeric part (e.g., "1" from "1", handle blanks)
    stats["SeedNum"] = pd.to_numeric(stats[config.SEED_COL], errors="coerce")

    # Select only columns we need
    keep_cols = (
        ["TeamID", config.STATS_SEASON_COL, "SeedNum"]
        + config.FEATURES
        + [config.STATS_TEAM_NAME_COL]
    )
    # Only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in stats.columns]
    stats = stats[keep_cols].copy()

    # Convert feature columns to numeric
    for col in config.FEATURES:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")

    return stats


def load_tournament_games() -> pd.DataFrame:
    """Load tournament game results."""
    games = pd.read_csv(config.TOURNEY_RESULTS_CSV)
    games = games[games["Season"].isin(config.TRAIN_SEASONS)].copy()
    print(f"  Loaded {len(games)} tournament games ({games['Season'].min()}-{games['Season'].max()})")
    return games


def build_training_data() -> pd.DataFrame:
    """
    Join tournament games with team stats to create the training dataset.
    
    Each game produces TWO rows:
      Row 1: TeamA = Winner, TeamB = Loser, Label = 1
      Row 2: TeamA = Loser,  TeamB = Winner, Label = 0
    
    This avoids positional bias (the model can't learn that "Team A always wins").
    """
    stats = load_team_stats()
    games = load_tournament_games()

    # Create a lookup: (TeamID, Season) -> stats dict
    stats_idx = stats.set_index(["TeamID", config.STATS_SEASON_COL])

    feature_cols = [c for c in config.FEATURES if c in stats.columns] + ["SeedNum"]

    rows = []
    skipped = 0

    for _, game in games.iterrows():
        season = int(game["Season"])
        w_id = int(game["WTeamID"])
        l_id = int(game["LTeamID"])

        # Look up stats for both teams
        try:
            w_stats = stats_idx.loc[(w_id, season)]
            l_stats = stats_idx.loc[(l_id, season)]
        except KeyError:
            skipped += 1
            continue

        # Handle case where lookup returns multiple rows (shouldn't happen)
        if isinstance(w_stats, pd.DataFrame):
            w_stats = w_stats.iloc[0]
        if isinstance(l_stats, pd.DataFrame):
            l_stats = l_stats.iloc[0]

        # Row 1: Team A = Winner (label=1)
        row1 = {"Season": season, "Label": 1}
        for col in feature_cols:
            if col in w_stats.index and col in l_stats.index:
                row1[f"A_{col}"] = w_stats[col]
                row1[f"B_{col}"] = l_stats[col]

        # Row 2: Team A = Loser (label=0)
        row2 = {"Season": season, "Label": 0}
        for col in feature_cols:
            if col in w_stats.index and col in l_stats.index:
                row2[f"A_{col}"] = l_stats[col]
                row2[f"B_{col}"] = w_stats[col]

        rows.append(row1)
        rows.append(row2)

    df = pd.DataFrame(rows)

    if skipped:
        print(f"  ⚠️  Skipped {skipped} games (missing team stats)")

    print(f"  Created {len(df)} training rows from {len(df)//2} games")
    return df


def save_training_data(df: pd.DataFrame) -> Path:
    """Save training data to CSV."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TRAINING_DATA_CSV, index=False)
    print(f"  Saved to {config.TRAINING_DATA_CSV}")
    return config.TRAINING_DATA_CSV


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building training data...")
    df = build_training_data()
    save_training_data(df)

    # Quick summary
    print(f"\n{'='*60}")
    print(f"Training Data Summary")
    print(f"{'='*60}")
    print(f"  Shape:    {df.shape}")
    print(f"  Seasons:  {sorted(df['Season'].unique())}")
    print(f"  Labels:   {df['Label'].value_counts().to_dict()}")
    print(f"  NaN cols: {df.columns[df.isna().any()].tolist()}")
