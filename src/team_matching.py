"""
Team name matching: map ESPN team names from team_stats.csv to Mania TeamIDs.

Strategy:
  1. Load MTeamSpellings.csv as a {lowercase_spelling: TeamID} dictionary
  2. For each unique ESPN name in team_stats.csv, lowercase and look up
  3. Apply manual overrides for the 7 known mismatches
  4. Output team_name_mapping.csv and print a coverage report
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def build_spellings_lookup() -> dict[str, int]:
    """Return {lowercase_spelling: TeamID} from MTeamSpellings.csv."""
    df = pd.read_csv(config.MTEAM_SPELLINGS_CSV)
    return {
        str(row["TeamNameSpelling"]).strip().lower(): int(row["TeamID"])
        for _, row in df.iterrows()
    }


def build_team_id_to_name() -> dict[int, str]:
    """Return {TeamID: TeamName} from MTeams.csv."""
    df = pd.read_csv(config.MTEAMS_CSV)
    return {int(row["TeamID"]): str(row["TeamName"]) for _, row in df.iterrows()}


def match_teams() -> pd.DataFrame:
    """
    Match every unique ESPN team name in team_stats.csv to a Mania TeamID.

    Returns a DataFrame with columns:
        ESPNName, TeamID, ManiaTeamName, MatchMethod
    """
    spellings = build_spellings_lookup()
    id_to_name = build_team_id_to_name()

    stats = pd.read_csv(config.TEAM_STATS_CSV, usecols=[config.STATS_TEAM_NAME_COL])
    espn_names = sorted(stats[config.STATS_TEAM_NAME_COL].dropna().unique())

    rows: list[dict] = []
    unmatched: list[str] = []

    for name in espn_names:
        key = name.strip().lower()

        # 1) Try manual override first
        if key in config.MANUAL_OVERRIDES:
            tid = config.MANUAL_OVERRIDES[key]
            rows.append(
                {
                    "ESPNName": name,
                    "TeamID": tid,
                    "ManiaTeamName": id_to_name.get(tid, "???"),
                    "MatchMethod": "manual_override",
                }
            )
            continue

        # 2) Direct lookup in spellings
        if key in spellings:
            tid = spellings[key]
            rows.append(
                {
                    "ESPNName": name,
                    "TeamID": tid,
                    "ManiaTeamName": id_to_name.get(tid, "???"),
                    "MatchMethod": "spellings_exact",
                }
            )
            continue

        unmatched.append(name)

    mapping = pd.DataFrame(rows)

    # Report
    total = len(espn_names)
    matched = len(rows)
    print(f"\n{'='*60}")
    print(f"Team Name Matching Report")
    print(f"{'='*60}")
    print(f"Total ESPN names:  {total}")
    print(f"Matched:           {matched}  ({matched/total*100:.1f}%)")
    print(f"Unmatched:         {len(unmatched)}")

    if unmatched:
        print(f"\nUnmatched teams:")
        for u in unmatched:
            print(f"  - {u}")
        print(
            "\nAdd these to config.MANUAL_OVERRIDES with the correct TeamID."
        )

    return mapping


def save_mapping(mapping: pd.DataFrame) -> Path:
    """Save the mapping to CSV and return the path."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(config.TEAM_MAPPING_CSV, index=False)
    print(f"\nSaved mapping to {config.TEAM_MAPPING_CSV}")
    return config.TEAM_MAPPING_CSV


def load_mapping() -> pd.DataFrame:
    """Load previously saved mapping."""
    return pd.read_csv(config.TEAM_MAPPING_CSV)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mapping = match_teams()
    save_mapping(mapping)

    # Quick validation: every tournament team should be in the mapping
    tourney = pd.read_csv(config.TOURNEY_RESULTS_CSV)
    tourney_ids = set(tourney["WTeamID"].unique()) | set(tourney["LTeamID"].unique())
    mapped_ids = set(mapping["TeamID"].unique())
    missing = tourney_ids - mapped_ids

    if missing:
        id_to_name = build_team_id_to_name()
        print(f"\n[!] {len(missing)} tournament teams have no mapping:")
        for tid in sorted(missing):
            print(f"  TeamID {tid}: {id_to_name.get(tid, '???')}")
    else:
        print(f"\n[OK] All {len(tourney_ids)} tournament teams are covered!")
