"""
Monte Carlo bracket simulator for March Madness.

This module:
  1. Takes a 68-team bracket definition (team name, seed, region)
  2. Uses predict.py's predict_matchup() to get win probabilities for every matchup
  3. Runs N Monte Carlo simulations to get each team's probability of reaching each round
  4. Generates optimized brackets using multiple strategies:
       - Greedy (maximize expected score)
       - Upset specials (follow high-upside underdogs deep)
       - Diverse variants (swapped Final Four / champion picks)

Scoring (NCAA.com standard):
  Round of 64:   1 pt
  Round of 32:   2 pts
  Sweet 16:      4 pts
  Elite Eight:   8 pts
  Final Four:   16 pts
  Championship: 32 pts

Usage:
  python src/simulate.py --season 2025 --n-sims 10000 --n-brackets 5
  python src/simulate.py --bracket-file bracket.json --n-sims 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data_prep import load_team_stats, build_training_data
from feature_engineering import prepare_features
from models import train_final_models
from predict import predict_matchup, load_models, save_models, get_tournament_teams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_POINTS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}
ROUND_NAMES  = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}

# Standard 4-region bracket structure: (region, seed_matchups)
REGIONS = ["East", "West", "South", "Midwest"]

# First Four play-in games (lowest seeds): handled separately if present
FIRST_FOUR_SEEDS = [11, 16]  # seeds that typically have play-in games

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Team:
    """Represents a tournament team."""
    def __init__(self, name: str, seed: int, region: str, stats: pd.Series):
        self.name = name
        self.seed = seed
        self.region = region
        self.stats = stats  # row from team_stats DataFrame

    def __repr__(self):
        return f"#{self.seed} {self.name} ({self.region})"


class Bracket:
    """
    Represents a filled-out bracket.
    Stores picks as a dict: round -> list of team names that win that round.
    """
    def __init__(self, strategy_name: str = ""):
        self.strategy_name = strategy_name
        # picks[round] = list of team names predicted to win that round
        self.picks: dict[int, list[str]] = {r: [] for r in range(1, 7)}
        self.expected_score: float = 0.0
        self.notes: list[str] = []

    def score_against(self, actual_results: dict[int, list[str]]) -> int:
        """Score this bracket against actual tournament results."""
        total = 0
        for round_num, winners in actual_results.items():
            pts = ROUND_POINTS[round_num]
            for team in self.picks.get(round_num, []):
                if team in winners:
                    total += pts
        return total

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "expected_score": round(self.expected_score, 2),
            "notes": self.notes,
            "picks": self.picks,
        }


# ---------------------------------------------------------------------------
# Probability cache
# ---------------------------------------------------------------------------

class MatchupCache:
    """Caches win probability lookups to avoid repeated model calls."""

    def __init__(self, models: dict, scaler, feature_cols: list[str]):
        self.models = models
        self.scaler = scaler
        self.feature_cols = feature_cols
        self._cache: dict[tuple, float] = {}

    def win_prob(self, team_a: Team, team_b: Team) -> float:
        """Return P(team_a beats team_b)."""
        key = (team_a.name, team_b.name)
        if key not in self._cache:
            result = predict_matchup(
                team_a.stats, team_b.stats,
                self.models, self.scaler, self.feature_cols
            )
            p = result["win_prob_a_logistic"]  # use best single model
            self._cache[key] = p
            self._cache[(team_b.name, team_a.name)] = 1.0 - p
        return self._cache[key]


# ---------------------------------------------------------------------------
# Bracket structure builder
# ---------------------------------------------------------------------------

def build_teams_from_stats(stats: pd.DataFrame, season: int) -> list[Team]:
    """Build Team objects from team_stats for a given season."""
    tourney = get_tournament_teams(stats, season)
    teams = []
    for _, row in tourney.iterrows():
        name = row.get(config.STATS_TEAM_NAME_COL, f"Team_{row['SeedNum']}")
        seed = int(row["SeedNum"])
        region = row.get("Region", "Unknown")
        teams.append(Team(name=name, seed=seed, region=region, stats=row))
    return teams


def group_teams_by_region(teams: list[Team]) -> dict[str, list[Team]]:
    """Group teams by region, sorted by seed."""
    regions: dict[str, list[Team]] = defaultdict(list)
    for t in teams:
        regions[t.region].append(t)
    for r in regions:
        regions[r].sort(key=lambda t: t.seed)
    return dict(regions)


def get_first_round_matchups(region_teams: list[Team]) -> list[tuple[Team, Team]]:
    """
    Pair teams into first-round matchups using standard NCAA seeding:
    1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    (bracket order matters for subsequent rounds)
    """
    seed_order = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]
    by_seed: dict[int, Team] = {t.seed: t for t in region_teams}
    matchups = []
    for s_a, s_b in seed_order:
        if s_a in by_seed and s_b in by_seed:
            matchups.append((by_seed[s_a], by_seed[s_b]))
    return matchups


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def simulate_tournament_once(
    region_matchups: dict[str, list[tuple[Team, Team]]],
    final_four_matchups: list[tuple[str, str]],  # (region_name_a, region_name_b)
    cache: MatchupCache,
    rng: random.Random,
) -> dict[int, list[str]]:
    """
    Simulate one full tournament. Returns dict: round -> list of winners.
    Regions: 4 regions x 4 rounds = rounds 1-4.
    Final Four = round 5, Championship = round 6.
    """
    results: dict[int, list[str]] = {r: [] for r in range(1, 7)}

    # region_bracket[region] = list of Teams advancing from each round
    region_survivors: dict[str, list[Team]] = {}

    for region, matchups in region_matchups.items():
        survivors = list(matchups)  # list of (Team, Team) pairs for round 1

        for round_num in range(1, 5):  # rounds 1-4 within region
            next_round: list[tuple[Team, Team]] = []
            round_winners: list[Team] = []

            for team_a, team_b in survivors:
                p = cache.win_prob(team_a, team_b)
                winner = team_a if rng.random() < p else team_b
                round_winners.append(winner)
                results[round_num].append(winner.name)

            # Pair winners for next round (bracket order preserved)
            for i in range(0, len(round_winners), 2):
                if i + 1 < len(round_winners):
                    next_round.append((round_winners[i], round_winners[i+1]))

            survivors = next_round

        # Regional champion = last survivor
        if round_winners:
            region_survivors[region] = round_winners[-1]

    # Final Four (round 5)
    final_four_teams: list[Team] = []
    for region_a, region_b in final_four_matchups:
        team_a = region_survivors.get(region_a)
        team_b = region_survivors.get(region_b)
        if team_a and team_b:
            p = cache.win_prob(team_a, team_b)
            winner = team_a if rng.random() < p else team_b
            final_four_teams.append(winner)
            results[5].append(winner.name)

    # Championship (round 6)
    if len(final_four_teams) == 2:
        team_a, team_b = final_four_teams
        p = cache.win_prob(team_a, team_b)
        winner = team_a if rng.random() < p else team_b
        results[6].append(winner.name)

    return results


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    region_matchups: dict[str, list[tuple[Team, Team]]],
    final_four_matchups: list[tuple[str, str]],
    cache: MatchupCache,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """
    Run n_sims simulations. Returns:
      reach_probs[team_name][round] = probability of reaching that round
    """
    rng = random.Random(seed)
    reach_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    print(f"  Running {n_sims:,} simulations...")
    for i in range(n_sims):
        sim_result = simulate_tournament_once(
            region_matchups, final_four_matchups, cache, rng
        )
        for round_num, winners in sim_result.items():
            for team in winners:
                reach_counts[team][round_num] += 1

    # Convert to probabilities
    reach_probs: dict[str, dict[int, float]] = {}
    for team, rounds in reach_counts.items():
        reach_probs[team] = {r: rounds[r] / n_sims for r in range(1, 7)}

    return reach_probs


# ---------------------------------------------------------------------------
# Expected score computation
# ---------------------------------------------------------------------------

def expected_score_for_pick(team_name: str, round_num: int,
                             reach_probs: dict[str, dict[int, float]]) -> float:
    """Expected points from picking team_name to win round_num."""
    p = reach_probs.get(team_name, {}).get(round_num, 0.0)
    return p * ROUND_POINTS[round_num]


# ---------------------------------------------------------------------------
# Bracket generation strategies
# ---------------------------------------------------------------------------

def pick_winner_by_ev(
    team_a: Team, team_b: Team, round_num: int,
    reach_probs: dict[str, dict[int, float]]
) -> Team:
    """Pick whoever has higher expected value for this round."""
    ev_a = expected_score_for_pick(team_a.name, round_num, reach_probs)
    ev_b = expected_score_for_pick(team_b.name, round_num, reach_probs)
    return team_a if ev_a >= ev_b else team_b


def generate_greedy_bracket(
    region_matchups: dict[str, list[tuple[Team, Team]]],
    final_four_matchups: list[tuple[str, str]],
    reach_probs: dict[str, dict[int, float]],
    all_teams_by_name: dict[str, Team],
    cache: MatchupCache,
    forced_champion: Optional[str] = None,
    forced_final_four: Optional[list[str]] = None,
) -> Bracket:
    """
    Generate a bracket by greedily picking the higher-EV team at each matchup.
    Optionally force a specific champion or Final Four picks.

    forced_final_four: list of team names that MUST win their region (reach F4).
    forced_champion: team name that MUST win the championship.
    If a forced team can't physically reach the forced slot (already eliminated
    by another forced pick or not in the bracket), falls back to EV.
    """
    bracket = Bracket()
    region_champions: dict[str, Team] = {}

    # Build a set of all forced deep teams (union of forced_final_four + champion)
    forced_deep: set[str] = set()
    if forced_final_four:
        forced_deep.update(forced_final_four)
    if forced_champion:
        forced_deep.add(forced_champion)

    for region, matchups in region_matchups.items():
        survivors = list(matchups)
        # Determine if a forced team lives in this region
        region_team_names = {t.name for pair in matchups for t in pair}
        forced_in_region = forced_deep & region_team_names

        for round_num in range(1, 5):
            next_round: list[tuple[Team, Team]] = []
            round_winners: list[Team] = []

            for team_a, team_b in survivors:
                # Force the pick if one of the forced teams is in this matchup
                if forced_in_region:
                    if team_a.name in forced_in_region:
                        winner = team_a
                    elif team_b.name in forced_in_region:
                        winner = team_b
                    else:
                        winner = pick_winner_by_ev(team_a, team_b, round_num, reach_probs)
                else:
                    winner = pick_winner_by_ev(team_a, team_b, round_num, reach_probs)

                round_winners.append(winner)
                bracket.picks[round_num].append(winner.name)

            for i in range(0, len(round_winners), 2):
                if i + 1 < len(round_winners):
                    next_round.append((round_winners[i], round_winners[i+1]))

            survivors = next_round

        if round_winners:
            region_champions[region] = round_winners[-1]

    # Final Four
    final_four_winners: list[Team] = []
    for region_a, region_b in final_four_matchups:
        team_a = region_champions.get(region_a)
        team_b = region_champions.get(region_b)
        if team_a and team_b:
            # Force the pick if champion must come through this semifinal
            if forced_champion and team_a.name == forced_champion:
                winner = team_a
            elif forced_champion and team_b.name == forced_champion:
                winner = team_b
            else:
                winner = pick_winner_by_ev(team_a, team_b, 5, reach_probs)
            final_four_winners.append(winner)
            bracket.picks[5].append(winner.name)

    # Championship
    if len(final_four_winners) == 2:
        team_a, team_b = final_four_winners
        if forced_champion and team_a.name == forced_champion:
            winner = team_a
        elif forced_champion and team_b.name == forced_champion:
            winner = team_b
        else:
            winner = pick_winner_by_ev(team_a, team_b, 6, reach_probs)
        bracket.picks[6].append(winner.name)

    # Compute expected score
    bracket.expected_score = sum(
        expected_score_for_pick(team, r, reach_probs)
        for r, teams in bracket.picks.items()
        for team in teams
    )

    return bracket


def get_top_upset_candidates(
    reach_probs: dict[str, dict[int, float]],
    all_teams_by_name: dict[str, Team],
    round_threshold: int = 4,  # Elite Eight and beyond
    min_seed: int = 6,         # Only consider seeds 6+
    top_n: int = 5,
) -> list[tuple[Team, float]]:
    """
    Find teams with meaningful deep-run probability despite high seed number.
    Returns list of (Team, probability_of_reaching_round_threshold).
    """
    candidates = []
    for name, probs in reach_probs.items():
        team = all_teams_by_name.get(name)
        if team and team.seed >= min_seed:
            deep_run_prob = probs.get(round_threshold, 0.0)
            if deep_run_prob > 0.05:  # at least 5% chance of Elite Eight
                candidates.append((team, deep_run_prob))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Main bracket generation
# ---------------------------------------------------------------------------

def generate_all_brackets(
    region_matchups: dict[str, list[tuple[Team, Team]]],
    final_four_matchups: list[tuple[str, str]],
    reach_probs: dict[str, dict[int, float]],
    all_teams_by_name: dict[str, Team],
    cache: MatchupCache,
    n_brackets: int = 5,
) -> list[Bracket]:
    """Generate n_brackets using different strategies."""
    brackets = []

    # --- Bracket 1: Pure greedy EV ---
    b1 = generate_greedy_bracket(
        region_matchups, final_four_matchups, reach_probs,
        all_teams_by_name, cache
    )
    b1.strategy_name = "Greedy EV (chalk)"
    b1.notes = ["Maximizes expected score at every matchup", "Best single bracket"]
    brackets.append(b1)

    if n_brackets < 2:
        return brackets

    # Find the top champion candidate (by championship probability)
    champ_probs = [(name, probs.get(6, 0.0)) for name, probs in reach_probs.items()]
    champ_probs.sort(key=lambda x: x[1], reverse=True)
    top_champs = [name for name, _ in champ_probs[:6]]

    # --- Bracket 2: Same as #1 but swap champion to #2 most likely ---
    if len(top_champs) >= 2 and top_champs[1] != b1.picks[6][0] if b1.picks[6] else True:
        alt_champ = top_champs[1] if top_champs[0] == (b1.picks[6][0] if b1.picks[6] else "") else top_champs[0]
        b2 = generate_greedy_bracket(
            region_matchups, final_four_matchups, reach_probs,
            all_teams_by_name, cache,
            forced_champion=alt_champ
        )
        b2.strategy_name = f"Alt Champion: {alt_champ}"
        b2.notes = [f"Same structure as greedy, champion swapped to {alt_champ}",
                    f"Champion probability: {reach_probs.get(alt_champ, {}).get(6, 0):.1%}"]
        brackets.append(b2)
    else:
        brackets.append(deepcopy(b1))

    if n_brackets < 3:
        return brackets

    # --- Bracket 3: Upset special — follow top upset candidate deep ---
    upset_candidates = get_upset_candidates_for_bracket(reach_probs, all_teams_by_name)
    if upset_candidates:
        top_upset = upset_candidates[0]
        b3 = generate_greedy_bracket(
            region_matchups, final_four_matchups, reach_probs,
            all_teams_by_name, cache,
            forced_final_four=[top_upset.name],
            forced_champion=top_upset.name
        )
        b3.strategy_name = f"Upset Special: {top_upset.name} (#{top_upset.seed})"
        b3.notes = [
            f"Built around {top_upset.name} (#{top_upset.seed} seed) making a deep run",
            f"Elite Eight probability: {reach_probs.get(top_upset.name, {}).get(4, 0):.1%}",
            f"Championship probability: {reach_probs.get(top_upset.name, {}).get(6, 0):.1%}",
        ]
        brackets.append(b3)
    else:
        brackets.append(deepcopy(b1))

    if n_brackets < 4:
        return brackets

    # --- Bracket 4: Second upset candidate or alt Final Four ---
    if len(upset_candidates) >= 2:
        upset2 = upset_candidates[1]
        b4 = generate_greedy_bracket(
            region_matchups, final_four_matchups, reach_probs,
            all_teams_by_name, cache,
            forced_final_four=[upset2.name]
        )
        b4.strategy_name = f"Upset Run: {upset2.name} (#{upset2.seed}) to Final Four"
        b4.notes = [
            f"{upset2.name} (#{upset2.seed}) goes deep but doesn't win it all",
            f"Final Four probability: {reach_probs.get(upset2.name, {}).get(5, 0):.1%}",
        ]
        brackets.append(b4)
    else:
        brackets.append(deepcopy(b1))

    if n_brackets < 5:
        return brackets

    # --- Bracket 5: Max divergence — pick #3 champion, second upset candidate ---
    alt_champ2 = top_champs[2] if len(top_champs) >= 3 else (top_champs[-1] if top_champs else None)
    b5 = generate_greedy_bracket(
        region_matchups, final_four_matchups, reach_probs,
        all_teams_by_name, cache,
        forced_champion=alt_champ2,
        forced_final_four=[upset_candidates[1].name] if len(upset_candidates) >= 2 else None
    )
    b5.strategy_name = f"Wildcard: {alt_champ2} wins, upset run included"
    b5.notes = [
        "Most divergent from bracket #1",
        f"Champion: {alt_champ2} ({reach_probs.get(alt_champ2, {}).get(6, 0):.1%} probability)" if alt_champ2 else "",
    ]
    brackets.append(b5)

    return brackets


def get_upset_candidates_for_bracket(
    reach_probs: dict[str, dict[int, float]],
    all_teams_by_name: dict[str, Team],
) -> list[Team]:
    """Return teams with seed >= 5 that have meaningful championship upside."""
    candidates = []
    for name, probs in reach_probs.items():
        team = all_teams_by_name.get(name)
        if team and team.seed >= 5:
            champ_prob = probs.get(6, 0.0)
            e8_prob = probs.get(4, 0.0)
            # Must have real upside: E8 > 8% or championship > 2%
            if e8_prob > 0.08 or champ_prob > 0.02:
                candidates.append((team, champ_prob, e8_prob))
    candidates.sort(key=lambda x: x[1] + x[2] * 0.3, reverse=True)
    return [t for t, _, _ in candidates]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_reach_probs(reach_probs: dict[str, dict[int, float]],
                      all_teams_by_name: dict[str, Team],
                      top_n: int = 20) -> None:
    """Print top teams by championship probability."""
    champ_probs = [
        (name, probs.get(6, 0.0), probs.get(5, 0.0), probs.get(4, 0.0),
         all_teams_by_name.get(name, Team("?", 0, "?", pd.Series())).seed)
        for name, probs in reach_probs.items()
    ]
    champ_probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'='*75}")
    print(f"  {'Team':<28} {'Seed':>4}  {'R64':>6} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Champ':>6}")
    print(f"  {'='*75}")
    for name, champ, f4, e8, seed in champ_probs[:top_n]:
        probs = reach_probs[name]
        print(
            f"  {name:<28} #{seed:<3}  "
            f"{probs.get(1,0):>5.1%} {probs.get(2,0):>6.1%} {probs.get(3,0):>6.1%} "
            f"{probs.get(4,0):>6.1%} {probs.get(5,0):>6.1%} {probs.get(6,0):>6.1%}"
        )


def print_bracket(bracket: Bracket) -> None:
    """Pretty-print a bracket's picks by round."""
    print(f"\n  Strategy: {bracket.strategy_name}")
    print(f"  Expected Score: {bracket.expected_score:.1f}")
    for note in bracket.notes:
        if note:
            print(f"  Note: {note}")
    print()
    for round_num in range(1, 7):
        teams = bracket.picks.get(round_num, [])
        label = ROUND_NAMES[round_num]
        print(f"    {label} ({ROUND_POINTS[round_num]}pt): {', '.join(teams) if teams else '—'}")


# ---------------------------------------------------------------------------
# Bracket file loader (for when actual bracket is known)
# ---------------------------------------------------------------------------

def load_bracket_file(path: str) -> dict:
    """
    Load a bracket definition from JSON. Format:
    {
      "season": 2026,
      "regions": {
        "East":    [{"name": "Duke",    "seed": 1}, {"name": "Vermont", "seed": 16}, ...],
        "West":    [...],
        "South":   [...],
        "Midwest": [...]
      },
      "final_four": [["East", "West"], ["South", "Midwest"]]
    }
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_region_matchups_from_stats(
    stats: pd.DataFrame, season: int
) -> tuple[dict[str, list[tuple[Team, Team]]], list[tuple[str, str]], dict[str, Team]]:
    """
    Auto-build region matchups from team_stats for a given season.
    Falls back gracefully if Region column is missing.
    """
    teams = build_teams_from_stats(stats, season)
    all_teams_by_name = {t.name: t for t in teams}

    # Check if Region column exists and is populated
    has_regions = any(t.region not in ("Unknown", "", None) for t in teams)

    if not has_regions:
        # No region data — distribute teams into 4 pseudo-regions by seed groups
        print("  ⚠️  No region data found. Distributing teams into pseudo-regions by seed.")
        pseudo_regions = ["East", "West", "South", "Midwest"]
        # Group all seed-1s into different regions, etc.
        by_seed: dict[int, list[Team]] = defaultdict(list)
        for t in teams:
            by_seed[t.seed].append(t)

        region_teams: dict[str, list[Team]] = {r: [] for r in pseudo_regions}
        for seed, seed_teams in sorted(by_seed.items()):
            for i, team in enumerate(seed_teams):
                region = pseudo_regions[i % 4]
                team.region = region
                region_teams[region].append(team)
    else:
        region_teams = group_teams_by_region(teams)

    region_matchups = {
        region: get_first_round_matchups(t_list)
        for region, t_list in region_teams.items()
    }

    # Standard Final Four pairings (East vs West, South vs Midwest)
    final_four_matchups = [("East", "West"), ("South", "Midwest")]

    return region_matchups, final_four_matchups, all_teams_by_name


def run(
    season: int,
    n_sims: int = 10000,
    n_brackets: int = 5,
    retrain: bool = False,
    bracket_file: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> list[Bracket]:
    """Full pipeline: load models → build bracket → simulate → generate brackets."""

    # Load or train models
    model_file = config.PROJECT_ROOT / "models" / "trained_models.pkl"
    if model_file.exists() and not retrain:
        print("Loading trained models...")
        models, scaler = load_models()
    else:
        print("Training models...")
        from data_prep import build_training_data
        training_df = build_training_data()
        X, y, seasons = prepare_features(training_df)
        models, scaler = train_final_models(X, y)
        save_models(models, scaler)

    stats = load_team_stats()
    feature_cols = config.FEATURES + ["SeedNum"]
    cache = MatchupCache(models, scaler, feature_cols)

    print(f"\nBuilding bracket structure for season {season}...")
    region_matchups, final_four_matchups, all_teams_by_name = \
        build_region_matchups_from_stats(stats, season)

    total_teams = sum(len(v) * 2 for v in region_matchups.values())
    print(f"  Teams in bracket: {total_teams}")
    for region, matchups in region_matchups.items():
        print(f"  {region}: {len(matchups)} first-round games")

    print(f"\nRunning Monte Carlo simulation ({n_sims:,} iterations)...")
    reach_probs = run_monte_carlo(
        region_matchups, final_four_matchups, cache, n_sims=n_sims
    )

    print(f"\nTop teams by championship probability:")
    print_reach_probs(reach_probs, all_teams_by_name)

    print(f"\nGenerating {n_brackets} optimized brackets...")
    brackets = generate_all_brackets(
        region_matchups, final_four_matchups, reach_probs,
        all_teams_by_name, cache, n_brackets=n_brackets
    )

    print(f"\n{'='*60}")
    print(f"  BRACKET RECOMMENDATIONS")
    print(f"{'='*60}")
    for i, bracket in enumerate(brackets, 1):
        print(f"\n  --- Bracket {i} ---")
        print_bracket(bracket)

    # Save outputs
    if output_dir is None:
        output_dir = config.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reach probabilities
    probs_records = []
    for name, probs in reach_probs.items():
        team = all_teams_by_name.get(name)
        seed = team.seed if team else 0
        region = team.region if team else "?"
        rec = {"team": name, "seed": seed, "region": region}
        for r in range(1, 7):
            rec[ROUND_NAMES[r]] = round(probs.get(r, 0.0), 4)
        probs_records.append(rec)

    probs_df = pd.DataFrame(probs_records).sort_values("R64", ascending=False)
    probs_path = output_dir / f"reach_probabilities_{season}.csv"
    probs_df.to_csv(probs_path, index=False)
    print(f"\n  Saved reach probabilities → {probs_path}")

    # Save brackets as JSON
    brackets_data = [b.to_dict() for b in brackets]
    brackets_path = output_dir / f"brackets_{season}.json"
    with open(brackets_path, "w") as f:
        json.dump(brackets_data, f, indent=2)
    print(f"  Saved brackets → {brackets_path}")

    return brackets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="March Madness Bracket Simulator")
    parser.add_argument("--season", type=int, default=None,
                        help="Season to simulate (default: latest in dataset)")
    parser.add_argument("--n-sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--n-brackets", type=int, default=5,
                        help="Number of brackets to generate (default: 5)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining")
    parser.add_argument("--bracket-file", type=str, default=None,
                        help="Path to bracket JSON file (for actual bracket input)")
    args = parser.parse_args()

    stats = load_team_stats()
    season = args.season or int(stats[config.STATS_SEASON_COL].max())

    run(
        season=season,
        n_sims=args.n_sims,
        n_brackets=args.n_brackets,
        retrain=args.retrain,
        bracket_file=args.bracket_file,
    )


if __name__ == "__main__":
    main()
