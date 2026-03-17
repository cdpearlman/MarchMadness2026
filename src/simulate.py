"""
Monte Carlo Tournament Simulator
Propagates per-game win probabilities through the bracket.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data_prep import load_team_stats
from predict import load_models, predict_matchup

def load_bracket(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def extract_teams_from_bracket(bracket: dict) -> list[str]:
    teams = []
    for region, seeds in bracket["regions"].items():
        for seed, team in seeds.items():
            if team.startswith("TBD_"):
                # Split First Four: TBD_UMBC_Howard -> UMBC, Howard
                parts = team[4:].split("_")
                teams.extend(parts)
            else:
                teams.append(team)
    return sorted(list(set(teams)))

def find_team_stat_row(stats: pd.DataFrame, team_name: str, season: int) -> pd.Series | None:
    mask = (stats[config.STATS_SEASON_COL] == season) & (stats[config.STATS_TEAM_NAME_COL] == team_name)
    matches = stats[mask]
    if len(matches) == 0:
        # Fallback to case-insensitive exact match
        mask = (stats[config.STATS_SEASON_COL] == season) & (stats[config.STATS_TEAM_NAME_COL].str.lower() == team_name.lower())
        matches = stats[mask]
        if len(matches) == 0:
            return None
    return matches.iloc[0]

def build_probability_cache(team_names: list[str], stats: pd.DataFrame, season: int, models: dict, scaler, feature_cols: list[str], calibrators) -> tuple[np.ndarray, dict[str, int]]:
    n_teams = len(team_names)
    team_to_idx = {name: i for i, name in enumerate(team_names)}
    idx_to_team = {i: name for i, name in enumerate(team_names)}
    
    # Store stats back for lookup
    team_rows = []
    for name in team_names:
        row = find_team_stat_row(stats, name, season)
        if row is None:
            print(f"  [X] Error: Could not find stats for team '{name}' in season {season}.")
            sys.exit(1)
        team_rows.append(row)
        
    prob_cache = np.zeros((n_teams, n_teams), dtype=np.float32)
    
    # Fill cache
    print(f"  Building probability cache for {n_teams} teams ({n_teams * (n_teams - 1) // 2} matchups)...")
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            result = predict_matchup(team_rows[i], team_rows[j], models, scaler, feature_cols, calibrators)
            win_prob_a = result["win_prob_a_ensemble"]
            prob_cache[i, j] = win_prob_a
            prob_cache[j, i] = 1.0 - win_prob_a
            
    return prob_cache, team_to_idx

def get_regions_order(bracket: dict) -> list[str]:
    """Flatten final_four_matchups to get consistent region ordering."""
    return [r for pair in bracket["final_four_matchups"] for r in pair]

def get_region_r1_matchups(region_seeds: dict, team_to_idx: dict) -> list:
    """Returns a list of tuples: ((seed_A, team_name_A_or_TBD), (seed_B, team_name_B_or_TBD)) in correct bracket order."""
    matchups = []
    for s1, s2 in config.MATCHUP_ORDER:
        t1 = region_seeds[str(s1)]
        t2 = region_seeds[str(s2)]
        matchups.append((t1, t2))
    return matchups

def resolve_first_four(n_sims: int, team1: str, team2: str, prob_cache: np.ndarray, team_to_idx: dict) -> np.ndarray:
    idx1 = team_to_idx[team1]
    idx2 = team_to_idx[team2]
    p1 = prob_cache[idx1, idx2]
    rand = np.random.random(n_sims).astype(np.float32)
    return np.where(rand < p1, idx1, idx2)

def _simulate_core(n_sims: int, bracket: dict, prob_cache: np.ndarray, team_to_idx: dict):
    """Core simulation — returns raw winner arrays for each round.

    Returns: (r1_winners, r2_winners, r3_winners, r4_winners, r5_winners, champ)
    where each rX_winners is a list of arrays of shape (n_sims,) and champ is a single array.
    """
    region_matchups = {}
    for region, seeds in bracket["regions"].items():
        region_matchups[region] = get_region_r1_matchups(seeds, team_to_idx)

    def get_team_array(team_str: str) -> np.ndarray:
        if team_str.startswith("TBD_"):
            parts = team_str[4:].split("_")
            return resolve_first_four(n_sims, parts[0], parts[1], prob_cache, team_to_idx)
        else:
            return np.full(n_sims, team_to_idx[team_str], dtype=np.int32)

    regions_order = get_regions_order(bracket)

    # R64: 4 regions x 8 games = 32 games
    r1_winners = []
    for r in regions_order:
        matchups = region_matchups[r]
        for t1, t2 in matchups:
            a_arr = get_team_array(t1)
            b_arr = get_team_array(t2)
            p_win_a = prob_cache[a_arr, b_arr]
            rand = np.random.random(n_sims).astype(np.float32)
            winners = np.where(rand < p_win_a, a_arr, b_arr)
            r1_winners.append(winners)

    def sim_round(prev_winners):
        next_winners = []
        for i in range(0, len(prev_winners), 2):
            a_arr = prev_winners[i]
            b_arr = prev_winners[i + 1]
            p_win_a = prob_cache[a_arr, b_arr]
            rand = np.random.random(n_sims).astype(np.float32)
            winners = np.where(rand < p_win_a, a_arr, b_arr)
            next_winners.append(winners)
        return next_winners

    r2_winners = sim_round(r1_winners)   # 16 games
    r3_winners = sim_round(r2_winners)   # 8 games (Sweet 16)
    r4_winners = sim_round(r3_winners)   # 4 games (Elite 8)
    r5_winners = sim_round(r4_winners)   # 2 games (Final Four)
    champ_arr = sim_round(r5_winners)    # 1 game (Championship)

    return r1_winners, r2_winners, r3_winners, r4_winners, r5_winners, champ_arr[0]


def simulate_bracket(n_sims: int, bracket: dict, prob_cache: np.ndarray, team_to_idx: dict):
    """Simulate bracket and return reach probability counts (original behavior)."""
    print(f"\nRunning {n_sims:,} simulations vectorized...")
    start_time = time.time()

    r1_winners, r2_winners, r3_winners, r4_winners, r5_winners, champ = \
        _simulate_core(n_sims, bracket, prob_cache, team_to_idx)

    elapsed = time.time() - start_time
    print(f"  Simulation complete in {elapsed:.2f}s")

    n_teams = prob_cache.shape[0]
    r1_counts = np.bincount(np.concatenate(r1_winners), minlength=n_teams)
    r2_counts = np.bincount(np.concatenate(r2_winners), minlength=n_teams)
    r3_counts = np.bincount(np.concatenate(r3_winners), minlength=n_teams)
    r4_counts = np.bincount(np.concatenate(r4_winners), minlength=n_teams)
    r5_counts = np.bincount(np.concatenate(r5_winners), minlength=n_teams)
    champ_counts = np.bincount(champ, minlength=n_teams)

    return {
        "p_r32": r1_counts / n_sims,
        "p_s16": r2_counts / n_sims,
        "p_e8": r3_counts / n_sims,
        "p_f4": r4_counts / n_sims,
        "p_final": r5_counts / n_sims,
        "p_champ": champ_counts / n_sims
    }


def simulate_bracket_raw(n_sims: int, bracket: dict, prob_cache: np.ndarray, team_to_idx: dict):
    """Simulate and return full game-by-game outcomes.

    Returns:
        sim_outcomes: (n_sims, 63) int16 — winner team index for each game slot
        round_points: (63,) float32 — ESPN point value for each slot
    """
    print(f"\nRunning {n_sims:,} raw simulations...")
    start_time = time.time()

    r1_w, r2_w, r3_w, r4_w, r5_w, champ = \
        _simulate_core(n_sims, bracket, prob_cache, team_to_idx)

    # Stack: 32 R64 + 16 R32 + 8 S16 + 4 E8 + 2 F4 + 1 Champ = 63 slots
    all_winners = r1_w + r2_w + r3_w + r4_w + r5_w + [champ]
    sim_outcomes = np.column_stack(all_winners).astype(np.int16)

    round_points = np.array(
        [1] * 32 + [2] * 16 + [4] * 8 + [8] * 4 + [16] * 2 + [32] * 1,
        dtype=np.float32
    )

    elapsed = time.time() - start_time
    print(f"  Raw simulation complete in {elapsed:.2f}s")

    return sim_outcomes, round_points

def main():
    parser = argparse.ArgumentParser(description="March Madness Monte Carlo Simulator")
    parser.add_argument("--n-sims", type=int, default=50000, help="Number of simulations to run")
    parser.add_argument("--season", type=int, default=2026, help="Season to simulate")
    parser.add_argument("--first-four-mode", choices=["favorite", "average", "simulate"], default="simulate", 
                        help="How to handle First Four games. Overridden by simulated vector approach.")
    parser.add_argument("--output", type=str, default=str(config.PROCESSED_DIR / "reach_probabilities_{}.csv"), help="Output path")
    args = parser.parse_args()

    # Create output dir
    output_path = Path(args.output.replace("{}", str(args.season)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading models and data...")
    models, scaler, calibrators = load_models()
    stats = load_team_stats()
    
    bracket_path = config.PROJECT_ROOT / "data" / f"bracket_{args.season}.json"
    if not bracket_path.exists():
        print(f"  [X] Bracket file {bracket_path} not found.")
        sys.exit(1)
        
    bracket = load_bracket(bracket_path)
    team_names = extract_teams_from_bracket(bracket)
    feature_cols = config.FEATURES

    prob_cache, team_to_idx = build_probability_cache(team_names, stats, args.season, models, scaler, feature_cols, calibrators)
    idx_to_team = {v: k for k, v in team_to_idx.items()}

    # Simulate
    results = simulate_bracket(args.n_sims, bracket, prob_cache, team_to_idx)
    
    # Generate Output Dataframe
    rows = []
    
    # Assign seeds and regions to teams
    team_seed_region = {}
    for region, seeds in bracket["regions"].items():
        for seed_str, team_str in seeds.items():
            seed = int(seed_str)
            if team_str.startswith("TBD_"):
                parts = team_str[4:].split("_")
                for p in parts:
                    team_seed_region[p] = (seed, region)
            else:
                team_seed_region[team_str] = (seed, region)
                
    for i in range(len(team_names)):
        t_name = idx_to_team[i]
        seed, region = team_seed_region[t_name]
        
        p_r32 = results["p_r32"][i]
        p_s16 = results["p_s16"][i]
        p_e8  = results["p_e8"][i]
        p_f4  = results["p_f4"][i]
        p_final = results["p_final"][i]
        p_champ = results["p_champ"][i]
        
        expected_score = (p_r32 * 1) + (p_s16 * 2) + (p_e8 * 4) + (p_f4 * 8) + (p_final * 16) + (p_champ * 32)
        
        rows.append({
            "team": t_name,
            "seed": seed,
            "region": region,
            "p_r32": p_r32,
            "p_s16": p_s16,
            "p_e8": p_e8,
            "p_f4": p_f4,
            "p_final": p_final,
            "p_champ": p_champ,
            "expected_score": expected_score
        })
        
    df = pd.DataFrame(rows)
    df = df.sort_values(by="p_champ", ascending=False).reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved reach probabilities to {output_path}")
    
    # Print Stdout Report
    print(f"\n{args.season} NCAA Tournament Probability Matrix ({args.n_sims:,} simulations)")
    print("═" * 84)
    print(f"{'Team':<20} {'Seed':<4} {'Region':<9} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Final':>6} {'Champ':>6} {'E[Pts]':>6}")
    print("─" * 84)
    
    for _, row in df.iterrows():
        # Hide teams with 0% chance of making the round of 64 completely if they are first four losers? 
        # For simplicity, print all 68, but we can stop at reasonable threshold or just print top 30
        print(f"{row['team']:<20} {row['seed']:<4} {row['region']:<9} "
              f"{row['p_r32']:6.1%} {row['p_s16']:6.1%} {row['p_e8']:6.1%} "
              f"{row['p_f4']:6.1%} {row['p_final']:6.1%} {row['p_champ']:6.1%} {row['expected_score']:6.1f}")
              
if __name__ == "__main__":
    main()
