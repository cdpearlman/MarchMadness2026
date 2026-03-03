"""
Ingest Barttorvik season CSVs into a single team_stats.csv.

Reads per-season CSVs from data/barttorvik/YYYY.csv (no headers),
assigns column names based on the barttorvik team-tables_each.php
40-column layout, and writes a combined team_stats.csv.

Column mapping derived from cbbdata R package source + manual verification.

Usage:
    python ingest_barttorvik.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
BARTTORVIK_DIR = PROJECT_ROOT / "data" / "barttorvik"
SEED_REF = PROJECT_ROOT / "data" / "seed_reference.csv"
OUTPUT_CSV = PROJECT_ROOT / "team_stats.csv"

# Column names for barttorvik team-tables_each.php CSV (40 columns).
# Positions 0-36 match the trank.php layout from cbbdata R package.
# Positions 37-39 are the extra columns from team-tables_each.
BARTTORVIK_COLUMNS = [
    "team",                 # 0  - Team name
    "adj_o",                # 1  - Adjusted offensive efficiency
    "adj_d",                # 2  - Adjusted defensive efficiency
    "barthag",              # 3  - Win prob vs average D1 team (neutral)
    "record",               # 4  - W-L record string
    "wins",                 # 5  - Win count
    "games",                # 6  - Games played
    "efg",                  # 7  - Effective FG% (offense)
    "def_efg",              # 8  - Effective FG% (defense)
    "ftr",                  # 9  - Free throw rate (offense) = FTA/FGA
    "def_ftr",              # 10 - Free throw rate (defense)
    "tov_rate",             # 11 - Turnover rate (offense)
    "def_tov_rate",         # 12 - Turnover rate forced (defense)
    "oreb_rate",            # 13 - Offensive rebound rate
    "dreb_rate",            # 14 - Defensive rebound rate (opp OR%)
    "raw_oe",               # 15 - Raw offensive efficiency (unadjusted)
    "two_pt_pct",           # 16 - 2-point FG% (offense)
    "def_two_pt_pct",       # 17 - 2-point FG% (defense)
    "three_pt_pct",         # 18 - 3-point FG% (offense)
    "def_three_pt_pct",     # 19 - 3-point FG% (defense)
    "block_rate",           # 20 - Block rate
    "block_rate_allowed",   # 21 - Block rate allowed (opponent blocks)
    "assist_rate",          # 22 - Assist rate
    "def_assist_rate",      # 23 - Assist rate (opponent)
    "three_fg_rate",        # 24 - 3-point attempt rate (offense)
    "def_three_fg_rate",    # 25 - 3-point attempt rate (defense)
    "adj_t",                # 26 - Adjusted tempo
    "raw_de",               # 27 - Raw defensive efficiency (unadjusted)
    "unk_28",               # 28 - Unknown (possibly percentile/rank)
    "unk_29",               # 29 - Unknown
    "year",                 # 30 - Season year
    "unk_31",               # 31 - Unknown
    "unk_32",               # 32 - Unknown
    "unk_33",               # 33 - Unknown
    "unk_34",               # 34 - Empty in pre-tournament filtered data
    "ft_pct",               # 35 - Free throw % (offense)
    "def_ft_pct",           # 36 - Free throw % (defense)
    "eff_height",           # 37 - Effective height
    "experience",           # 38 - Team experience
    "unk_39",               # 39 - Unknown (possibly SOS-related)
]


def load_season_csv(path: Path) -> pd.DataFrame:
    """Load a single barttorvik season CSV (no header row)."""
    df = pd.read_csv(path, header=None, encoding="utf-8-sig")

    if len(df.columns) != len(BARTTORVIK_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(BARTTORVIK_COLUMNS)} columns, "
            f"got {len(df.columns)}"
        )

    df.columns = BARTTORVIK_COLUMNS
    return df


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns the pipeline needs."""
    df["adj_em"] = pd.to_numeric(df["adj_o"], errors="coerce") - pd.to_numeric(
        df["adj_d"], errors="coerce"
    )
    return df


def merge_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Merge tournament seeds from the reference file."""
    if not SEED_REF.exists():
        print(f"  Warning: {SEED_REF} not found, Seed column will be empty")
        df["Seed"] = pd.NA
        return df

    seeds = pd.read_csv(SEED_REF)
    # seed_reference has (TeamID, Season, SeedNum) — we need to match by team
    # Since we don't have TeamIDs yet (team_matching hasn't run), we'll
    # populate seeds AFTER team_matching. For now, add an empty column.
    df["Seed"] = pd.NA
    return df


def ingest() -> pd.DataFrame:
    """Read all season CSVs, combine, validate, and write output."""
    csv_files = sorted(BARTTORVIK_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {BARTTORVIK_DIR}")

    print(f"Found {len(csv_files)} season files in {BARTTORVIK_DIR}")

    frames = []
    for path in csv_files:
        season = int(path.stem)
        df = load_season_csv(path)

        # Use embedded year column, fallback to filename
        if "year" in df.columns:
            years_in_data = df["year"].dropna().unique()
            if len(years_in_data) == 1 and int(years_in_data[0]) != season:
                print(f"  Warning: {path.name} year column says "
                      f"{int(years_in_data[0])}, filename says {season}")

        df["Season"] = season
        frames.append(df)
        print(f"  {path.name}: {len(df)} teams")

    combined = pd.concat(frames, ignore_index=True)
    combined = derive_columns(combined)
    combined = merge_seeds(combined)

    # Validate
    for col in ["adj_o", "adj_d", "adj_t", "efg", "team"]:
        nulls = combined[col].isna().sum()
        if nulls > 0:
            print(f"  Warning: {nulls} nulls in '{col}'")

    # Write
    combined.to_csv(OUTPUT_CSV, index=False)
    seasons = sorted(combined["Season"].unique())
    print(f"\nWrote {len(combined)} rows ({len(seasons)} seasons) to {OUTPUT_CSV}")
    print(f"Seasons: {seasons[0]}-{seasons[-1]}")

    return combined


if __name__ == "__main__":
    ingest()
