"""
Feature engineering: compute stat differentials from raw team stats.

For every feature F, we create:
    diff_F = A_F - B_F

This captures relative team strength, which is what matters for predicting
head-to-head outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def compute_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a training DataFrame with columns A_<feat> and B_<feat>,
    compute diff_<feat> = A_<feat> - B_<feat> for all features.

    Returns a DataFrame with:
      - Season, Label
      - diff_<feature> columns
    """
    feature_cols = [c for c in config.FEATURES if f"A_{c}" in df.columns]

    # Include SeedNum if present (it's added by data_prep but not in config.FEATURES)
    if "A_SeedNum" in df.columns:
        feature_cols.append("SeedNum")

    result = df[["Season", "Label"]].copy()

    for col in feature_cols:
        a_col = f"A_{col}"
        b_col = f"B_{col}"
        if a_col in df.columns and b_col in df.columns:
            result[f"diff_{col}"] = df[a_col].astype(float) - df[b_col].astype(float)

    return result


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the list of diff_* column names used as model input."""
    return [c for c in df.columns if c.startswith("diff_")]


def handle_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in feature columns.

    Physical features (EffectiveHeight, Experience) are only available from
    2007+. We fill NaN with 0 for differentials (i.e., assume 'no advantage').
    """
    diff_cols = get_feature_names(df)
    n_missing_before = df[diff_cols].isna().sum().sum()

    # Fill NaN differentials with 0 (no advantage assumed)
    df[diff_cols] = df[diff_cols].fillna(0.0)

    n_missing_after = df[diff_cols].isna().sum().sum()
    if n_missing_before > 0:
        print(f"  Filled {n_missing_before} NaN differential values with 0")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Full feature pipeline: differentials → handle missing → return X, y, seasons.

    Returns:
        X: DataFrame of diff_* features
        y: Series of labels (0/1)
        seasons: Series of season values (for LOSO CV)
    """
    diff_df = compute_differentials(df)
    diff_df = handle_missing_features(diff_df)

    feature_names = get_feature_names(diff_df)
    X = diff_df[feature_names]
    y = diff_df["Label"]
    seasons = diff_df["Season"]

    print(f"\n{'='*60}")
    print(f"Feature Engineering Summary")
    print(f"{'='*60}")
    print(f"  Features:  {len(feature_names)}")
    print(f"  Samples:   {len(X)}")
    print(f"  Label dist: {y.value_counts().to_dict()}")
    print(f"  Feature names: {feature_names}")

    return X, y, seasons


# ---------------------------------------------------------------------------
# CLI entry point — runs the full pipeline from raw data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_prep import build_training_data

    print("Step 1: Building training data...")
    training_df = build_training_data()

    print("\nStep 2: Computing differentials...")
    X, y, seasons = prepare_features(training_df)

    print(f"\n  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"\n  Feature statistics:")
    print(X.describe().round(3).to_string())
