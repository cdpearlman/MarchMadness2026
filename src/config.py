"""
Configuration for March Madness predictive model.
Contains file paths, feature definitions, and team name overrides.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEAM_STATS_CSV = DATA_DIR / "team_stats.csv"
TOURNEY_RESULTS_CSV = DATA_DIR / "MNCAATourneyDetailedResults.csv"
MTEAMS_CSV = DATA_DIR / "MTeams.csv"
MTEAM_SPELLINGS_CSV = DATA_DIR / "MTeamSpellings.csv"

TEAM_MAPPING_CSV = PROCESSED_DIR / "team_name_mapping.csv"
TRAINING_DATA_CSV = PROCESSED_DIR / "training_data.csv"

# ---------------------------------------------------------------------------
# Team-name column in team_stats.csv used for matching
# ---------------------------------------------------------------------------
STATS_TEAM_NAME_COL = "Mapped ESPN Team Name"
STATS_SEASON_COL = "Season"

# ---------------------------------------------------------------------------
# Manual overrides: ESPN name (lowercase) -> TeamID
# These are the 7 names that don't match via MTeamSpellings.csv
# ---------------------------------------------------------------------------
MANUAL_OVERRIDES: dict[str, int] = {
    "app state": 1111,          # Appalachian St
    "miami": 1274,              # Miami FL (not Miami OH = 1275)
    "queens university": 1361,  # Queens NC
    "st. thomas-minnesota": 1384,  # St Thomas MN
    "ualbany": 1107,            # SUNY Albany
    "ul monroe": 1349,          # UL Monroe
    "ut rio grande valley": 1396,  # UT Rio Grande Valley
}

# ---------------------------------------------------------------------------
# Features to use (exact column names from team_stats.csv)
# All will be converted to differentials (TeamA - TeamB).
# Using Pre-Tournament variants where available to avoid data leakage.
# ---------------------------------------------------------------------------
FEATURES_PRE_TOURNAMENT = [
    # --- Adjusted efficiency (pre-tournament) ---
    "Pre-Tournament.AdjOE",
    "Pre-Tournament.AdjDE",
    "Pre-Tournament.AdjTempo",
    "Pre-Tournament.AdjEM",
]

FEATURES_FOUR_FACTORS_OFFENSE = [
    "eFGPct",       # Effective field goal %
    "TOPct",        # Turnover %
    "ORPct",        # Offensive rebound %
    "FTRate",       # Free-throw rate
]

FEATURES_SHOOTING = [
    "FG2Pct",       # 2-point FG %
    "FG3Pct",       # 3-point FG %
    "FTPct",        # Free-throw %
    "FG3Rate",      # 3-point attempt rate
]

FEATURES_OPPONENT = [
    "OppFG2Pct",    # Opponent 2-point FG %
    "OppFG3Pct",    # Opponent 3-point FG %
    "OppFTPct",     # Opponent FT %
    "OppFG3Rate",   # Opponent 3-point attempt rate
    "OppBlockPct",  # Opponent block %
    "OppStlRate",   # Opponent steal rate (how often they steal from us)
]

FEATURES_MISCELLANEOUS = [
    "BlockPct",     # Block %
    "StlRate",      # Steal rate
    "Net Rating",   # Net rating
]

FEATURES_PHYSICAL = [
    "EffectiveHeight",  # Available from 2007+
    "Experience",       # Available from 2007+
]

# Combined feature list (order preserved)
FEATURES: list[str] = (
    FEATURES_PRE_TOURNAMENT
    + FEATURES_FOUR_FACTORS_OFFENSE
    + FEATURES_SHOOTING
    + FEATURES_OPPONENT
    + FEATURES_MISCELLANEOUS
    + FEATURES_PHYSICAL
)

# Seed is treated separately (comes from team_stats.csv "Seed" column)
SEED_COL = "Seed"

# ---------------------------------------------------------------------------
# Model hyperparameters (defaults — tuned via CV)
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    use_label_encoder=False,
)

LOGISTIC_PARAMS = dict(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)

RANDOM_FOREST_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    random_state=42,
)

# Ensemble weights (LogReg, XGBoost, RF) — will be optimised via CV
ENSEMBLE_WEIGHTS = [0.30, 0.45, 0.25]

# Seasons to include in training (overlap of stats + tourney results)
TRAIN_SEASONS = list(range(2003, 2025))  # 2003-2024 inclusive
# 2020 had no tournament
TRAIN_SEASONS.remove(2020)

# Physical features only available from 2007
PHYSICAL_FEATURE_START_SEASON = 2007
