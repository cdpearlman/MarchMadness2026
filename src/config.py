"""
Configuration for March Madness predictive model.
Contains file paths, feature definitions, and team name overrides.

Data source: Barttorvik (barttorvik.com) team-tables_each.php CSVs.
Column names are native Barttorvik names from the 40-column CSV export.
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
SEED_REFERENCE_CSV = DATA_DIR / "data" / "seed_reference.csv"

TEAM_MAPPING_CSV = PROCESSED_DIR / "team_name_mapping.csv"
TRAINING_DATA_CSV = PROCESSED_DIR / "training_data.csv"

# ---------------------------------------------------------------------------
# Team-name column in team_stats.csv used for matching
# ---------------------------------------------------------------------------
STATS_TEAM_NAME_COL = "team"
STATS_SEASON_COL = "Season"

# ---------------------------------------------------------------------------
# Manual overrides: Barttorvik team name (lowercase) -> TeamID
# Populated after running team_matching.py on barttorvik data
# ---------------------------------------------------------------------------
MANUAL_OVERRIDES: dict[str, int] = {
    # Carried over from KenPom era (may still appear)
    "app state": 1111,              # Appalachian St
    "miami": 1274,                  # Miami FL (not Miami OH = 1275)
    "ualbany": 1107,                # SUNY Albany
    "ut rio grande valley": 1396,   # UT Rio Grande Valley
    # Barttorvik name mismatches
    "arkansas pine bluff": 1115,
    "bethune cookman": 1126,
    "cal st. bakersfield": 1167,
    "illinois chicago": 1227,       # IL Chicago
    "louisiana monroe": 1419,       # ULM
    "mississippi valley st.": 1290,
    "queens": 1474,                 # Queens NC
    "saint francis": 1384,          # Saint Francis PA
    "southeast missouri st.": 1369,
    "st. francis ny": 1383,
    "st. thomas-minnesota": 1384,   # St Thomas MN
    "tarleton st.": 1470,
    "tennessee martin": 1404,       # TN Martin
    "texas a&m corpus chris": 1394,
    "winston salem st.": 1445,
}

# ---------------------------------------------------------------------------
# Features to use (exact Barttorvik column names from team_stats.csv)
# All will be converted to differentials (TeamA - TeamB).
# Pre-tournament filtering is done at download time (end date = Selection Sunday).
# ---------------------------------------------------------------------------
FEATURES_ADJUSTED = [
    "adj_o",            # Adjusted offensive efficiency
    "adj_d",            # Adjusted defensive efficiency
    "adj_t",            # Adjusted tempo
    "adj_em",           # Adjusted efficiency margin (derived: adj_o - adj_d)
]

FEATURES_FOUR_FACTORS_OFFENSE = [
    "efg",              # Effective FG% (offense)
    "tov_rate",         # Turnover rate (offense)
    "oreb_rate",        # Offensive rebound rate
    "ftr",              # Free throw rate (offense) = FTA/FGA
]

FEATURES_SHOOTING = [
    "two_pt_pct",       # 2-point FG%
    "three_pt_pct",     # 3-point FG%
    "ft_pct",           # Free throw %
    "three_fg_rate",    # 3-point attempt rate
]

FEATURES_OPPONENT = [
    "def_two_pt_pct",   # Opponent 2-point FG%
    "def_three_pt_pct", # Opponent 3-point FG%
    "def_ft_pct",       # Opponent FT%
    "def_three_fg_rate",# Opponent 3-point attempt rate
    "block_rate_allowed",  # Opponent block rate
]

FEATURES_DEFENSE = [
    "def_efg",          # Defensive eFG% allowed
    "def_tov_rate",     # Defensive turnover rate forced
    "dreb_rate",        # Defensive rebound rate (opponent OR%)
    "def_ftr",          # Defensive free throw rate allowed
]

FEATURES_MISCELLANEOUS = [
    "block_rate",       # Block rate
    "barthag",          # Win probability vs average D1 team (replaces Net Rating)
]

FEATURES_PHYSICAL = [
    "eff_height",       # Effective height (available all Barttorvik seasons)
    "experience",       # Team experience (available all Barttorvik seasons)
]

# Combined feature list (order preserved)
FEATURES: list[str] = (
    FEATURES_ADJUSTED
    + FEATURES_FOUR_FACTORS_OFFENSE
    + FEATURES_SHOOTING
    + FEATURES_OPPONENT
    + FEATURES_DEFENSE
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

# Ensemble weights (LogReg, XGBoost, RF) — optimized via LOSO CV log-loss
ENSEMBLE_WEIGHTS = [0.5139, 0.0, 0.4861]

# Seasons to include in training (Barttorvik data starts 2008)
TRAIN_SEASONS = list(range(2008, 2026))  # 2008-2025 inclusive
# 2020 had no tournament
TRAIN_SEASONS.remove(2020)

# Physical features available for all Barttorvik seasons (2008+)
PHYSICAL_FEATURE_START_SEASON = 2008

# ---------------------------------------------------------------------------
# Bracket generation (champion x temperature stratified sampling)
# ---------------------------------------------------------------------------
BRACKET_N_CHAMPIONS = 5
BRACKET_TEMPERATURE_TIERS = [0.4, 0.9, 1.6]
BRACKET_CANDIDATES_PER_CELL = 12
