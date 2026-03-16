"""
Bracket Generation Module

Generates EV-optimized brackets for March Madness using model probabilities
and public ownership data. Supports purely chalk, value, and contrarian modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data_prep import load_team_stats
from predict import load_models, predict_matchup
from simulate import load_bracket, extract_teams_from_bracket, build_probability_cache

STANDARD_SCORING = {"r64": 1, "r32": 2, "s16": 4, "e8": 8, "f4": 16, "champ": 32}

def get_uniform_ownership(round_name: str) -> float:
    """Fallback uniform ownership if data is missing."""
    teams_alive = {"r64": 32, "r32": 16, "s16": 8, "e8": 4, "f4": 2, "champ": 1}
    return 1.0 / teams_alive.get(round_name, 64)

def load_ownership(path: str | Path) -> dict:
    if not path or not Path(path).exists():
        print(f"  [!] Warning: Ownership file {path} not found. Using uniform fallback.")
        return {}
    with open(path, "r") as f:
        data = json.load(f)
        if "UPDATE WITH REAL OWNERSHIP DATA" in data.get("note", ""):
            print("  [!] WARNING: Using placeholder ownership data from JSON note.")
        return data

def resolve_first_four(tbd_str: str, prob_cache: np.ndarray, team_to_idx: dict) -> str:
    """Resolve a TBD_TeamA_TeamB string by picking the model favorite."""
    parts = tbd_str[4:].split("_")
    if len(parts) != 2:
        # Fallback: if parsing fails, return first part
        print(f"  [!] Could not parse First Four entry '{tbd_str}', using '{parts[0]}'")
        return parts[0]
    t1, t2 = parts[0], parts[1]
    if t1 not in team_to_idx or t2 not in team_to_idx:
        # One team missing from stats — pick the one we have
        if t1 in team_to_idx:
            return t1
        if t2 in team_to_idx:
            return t2
        print(f"  [!] Neither First Four team found in cache: {t1}, {t2}")
        return t1
    p1 = prob_cache[team_to_idx[t1], team_to_idx[t2]]
    winner = t1 if p1 >= 0.5 else t2
    print(f"  First Four: {t1} vs {t2} -> {winner} ({max(p1, 1-p1):.1%})")
    return winner

def build_region_round1(region_seeds: dict, prob_cache: np.ndarray, team_to_idx: dict) -> list[tuple[str, str]]:
    """Get the standard R64 matchups for a region, resolving First Four via model."""
    order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    matchups = []
    for s1, s2 in order:
        t1, t2 = region_seeds[str(s1)], region_seeds[str(s2)]
        if t1.startswith("TBD_"): t1 = resolve_first_four(t1, prob_cache, team_to_idx)
        if t2.startswith("TBD_"): t2 = resolve_first_four(t2, prob_cache, team_to_idx)
        matchups.append((t1, t2))
    return matchups

def get_p_win(t1: str, t2: str, prob_cache: np.ndarray, team_to_idx: dict) -> float:
    """Get probability that t1 beats t2."""
    i1, i2 = team_to_idx[t1], team_to_idx[t2]
    return prob_cache[i1, i2]

def validate_upset(t1: str, t2: str, p1: float, round_name: str, bracket_type: str, ownership: dict) -> str | None:
    """Returns the underdog team if conditions are met, otherwise None."""
    p_underdog = 1 - p1 if p1 > 0.5 else p1
    model_underdog = t2 if p1 > 0.5 else t1
    
    thresholds = {
        "r64": {"p_min": 0.40, "own_max": 0.15},
        "r32": {"p_min": 0.38, "own_max": 0.30},
        "s16": {"p_min": 0.35, "own_max": 0.25},
        "e8":  {"p_min": 0.35, "own_max": 0.20},
    }
    
    if bracket_type == "contrarian":
        thresholds["s16"]["p_min"] = 0.30
        thresholds["e8"]["p_min"] = 0.30
        
    t = thresholds.get(round_name)
    if not t:
        return None
        
    if p_underdog >= t["p_min"]:
        u_own = ownership.get(round_name, {}).get(model_underdog)
        if u_own is None:
            u_own = get_uniform_ownership(round_name)
        
        if u_own <= t["own_max"]:
            return model_underdog
            
    return None

def generate_bracket(
    bracket_id: int,
    b_type: str,
    reach_probs: pd.DataFrame,
    ownership: dict,
    base_bracket: dict,
    prob_cache: np.ndarray,
    team_to_idx: dict,
    prev_brackets: list[dict]
) -> dict:
    
    # 1. Determine Champion
    if b_type == "chalk":
        champ = reach_probs.iloc[0]["team"]
        champ_value = reach_probs.iloc[0]["p_champ"] / (ownership.get("champion", {}).get(champ) or get_uniform_ownership("champ"))
    else:
        # compute value scores
        vals = []
        for _, row in reach_probs.iterrows():
            t = row["team"]
            own = ownership.get("champion", {}).get(t) or get_uniform_ownership("champ")
            val = row["p_champ"] / own
            vals.append((val, t))
        vals.sort(reverse=True)
        
        # Pick highest value champion not aggressively overlapping
        used_champs = [b["champion"] for b in prev_brackets]
        champ = None
        for val, t in vals:
            if t not in used_champs:
                champ = t
                champ_value = val
                break
        if not champ:
            # Fallback if somehow all used
            champ, champ_value = vals[-1][1], vals[-1][0]
            
    # Record locked teams
    locked_teams = {champ: ["r64", "r32", "s16", "e8", "f4", "champ"]}
    
    # Setup Final Four locks to support Champion cascade
    # For Champion to make it, they MUST win their region and their FF match
    # Find F4 teams
    region_of_champ = reach_probs[reach_probs["team"] == champ].iloc[0]["region"]
    ff_regions = [r for match in base_bracket["final_four_matchups"] for r in match]
    
    f4_teams = [champ]
    if b_type == "chalk":
        # just pick highest p_f4 in each remaining region
        for r in ff_regions:
            if r == region_of_champ: continue
            r_teams = reach_probs[reach_probs["region"] == r].sort_values("p_f4", ascending=False)
            t = r_teams.iloc[0]["team"]
            f4_teams.append(t)
            locked_teams[t] = ["r64", "r32", "s16", "e8"]
    else:
        # Build candidates for each non-champion region
        other_regions = [r for r in ff_regions if r != region_of_champ]
        region_candidates = {}
        for r in other_regions:
            r_teams = reach_probs[reach_probs["region"] == r].copy()
            r_teams["f4_own"] = r_teams["team"].apply(lambda t: ownership.get("final_four", {}).get(t) or get_uniform_ownership("f4"))
            r_teams["f4_val"] = r_teams["p_f4"] / r_teams["f4_own"]
            chalk_pick = r_teams.sort_values("p_f4", ascending=False).iloc[0]["team"]
            value_pick = r_teams.sort_values("f4_val", ascending=False).iloc[0]["team"]
            region_candidates[r] = {"chalk": chalk_pick, "value": value_pick, "r_teams": r_teams}

        if b_type == "contrarian":
            # All value picks
            for r in other_regions:
                t = region_candidates[r]["value"]
                f4_teams.append(t)
                locked_teams[t] = ["r64", "r32", "s16", "e8"]
        else:
            # Type 2: 2 chalk + 1 value pick
            # Find the region with the best value differential (value != chalk and high f4_val)
            best_value_region = None
            best_value_score = -1
            for r in other_regions:
                cand = region_candidates[r]
                if cand["value"] != cand["chalk"]:
                    val_row = cand["r_teams"][cand["r_teams"]["team"] == cand["value"]].iloc[0]
                    if val_row["f4_val"] > best_value_score:
                        best_value_score = val_row["f4_val"]
                        best_value_region = r

            for r in other_regions:
                if r == best_value_region:
                    t = region_candidates[r]["value"]
                else:
                    t = region_candidates[r]["chalk"]
                f4_teams.append(t)
                locked_teams[t] = ["r64", "r32", "s16", "e8"]

    # Now play the games
    regions_order = ff_regions # same as final four order
    
    picks = {"r64": [], "r32": [], "s16": [], "e8": [], "f4": [], "champ": [champ]}
    upsets = []
    
    # Play regional games
    current_round = []
    for r in regions_order:
        current_round.extend(build_region_round1(base_bracket["regions"][r], prob_cache, team_to_idx))
        
    rounds = ["r64", "r32", "s16", "e8", "f4"]
    for rd in rounds:
        next_round = []
        for i in range(len(current_round)):
            m = current_round[i]
            t1, t2 = m[0], m[1]
                
            p1 = get_p_win(t1, t2, prob_cache, team_to_idx)
            
            winner = None
            if t1 in locked_teams and rd in locked_teams[t1]: winner = t1
            elif t2 in locked_teams and rd in locked_teams[t2]: winner = t2
            else:
                if b_type != "chalk":
                    upset = validate_upset(t1, t2, p1, rd, b_type, ownership)
                    if upset:
                        winner = upset
                        u_p = 1 - p1 if p1 > 0.5 else p1
                        u_own = ownership.get(rd, {}).get(upset) or get_uniform_ownership(rd)
                        upsets.append({"round": rd, "game": f"{t1} vs {t2}", "pick": upset, "model_prob": float(u_p), "public_ownership": float(u_own)})
                
                if not winner:
                    winner = t1 if p1 > 0.5 else t2
            
            picks[rd].append(winner)
            next_round.append(winner)
            
        if rd != "f4":
            current_round = [(next_round[i], next_round[i+1]) for i in range(0, len(next_round), 2)]
            
    picks["f4_teams"] = f4_teams
            
    # Calculate Expected Score
    prob_col_map = {"r64": "p_r32", "r32": "p_s16", "s16": "p_e8", "e8": "p_f4", "f4": "p_final", "champ": "p_champ"}
    exp_score = 0.0
    for round_name, pts in STANDARD_SCORING.items():
        for t in picks[round_name]:
            t_row = reach_probs[reach_probs["team"] == t].iloc[0]
            exp_score += t_row[prob_col_map[round_name]] * pts
            
    bracket = {
        "bracket_id": bracket_id,
        "type": b_type,
        "champion": champ,
        "final_four": f4_teams,
        "elite_eight": picks["e8"],
        "sweet_16": picks["s16"],
        "r32": picks["r32"],
        "r64": picks["r64"],
        "expected_score": float(exp_score),
        "value_score_champion": float(champ_value),
        "upset_picks": upsets,
        "overlap_with_previous": None
    }
    
    # Calculate overlap against all previous brackets
    if prev_brackets:
        overlaps = []
        for prev in prev_brackets:
            overlaps.append(compute_bracket_overlap(bracket, prev))
        bracket["overlap_with_previous"] = float(max(overlaps))

    return bracket


def compute_bracket_overlap(a: dict, b: dict) -> float:
    """Compute fraction of identical picks between two brackets (position-aware)."""
    round_keys = ["r64", "r32", "sweet_16", "elite_eight", "final_four", "champion"]
    matches = 0
    total = 0
    for rd in round_keys:
        v1 = a[rd] if isinstance(a[rd], list) else [a[rd]]
        v2 = b[rd] if isinstance(b[rd], list) else [b[rd]]
        for pick1, pick2 in zip(v1, v2):
            total += 1
            if pick1 == pick2:
                matches += 1
    return matches / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Bracket Generator")
    parser.add_argument("--ownership", type=str, default="", help="Path to ownership JSON")
    parser.add_argument("--types", nargs="+", default=["chalk", "value", "contrarian"], help="Bracket types to generate")
    parser.add_argument("--no-ownership", action="store_true", help="Ignore ownership data")
    parser.add_argument("--output", type=str, default=str(config.PROCESSED_DIR / "brackets_{}.json"), help="Output path")
    parser.add_argument("--season", type=int, default=2026, help="Season to generate")
    args = parser.parse_args()

    out_path = Path(args.output.replace("{}", str(args.season)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading models, probs, and data...")
    models, scaler, calibrators = load_models()
    stats = load_team_stats()
    
    bracket_path = config.PROJECT_ROOT / "data" / f"bracket_{args.season}.json"
    if not bracket_path.exists():
        print(f"  [X] Bracket file {bracket_path} not found.")
        sys.exit(1)
        
    base_bracket = load_bracket(bracket_path)
    team_names = extract_teams_from_bracket(base_bracket)
    prob_cache, team_to_idx = build_probability_cache(team_names, stats, args.season, models, scaler, config.FEATURES, calibrators)
    
    prob_csv_path = config.PROCESSED_DIR / f"reach_probabilities_{args.season}.csv"
    if not prob_csv_path.exists():
        print(f"  [X] Prob matrix {prob_csv_path} not found. Run simulate.py first.")
        sys.exit(1)
    reach_probs = pd.read_csv(prob_csv_path)
    
    ownership = {}
    if not args.no_ownership and args.ownership:
        ownership = load_ownership(args.ownership)
        
    brackets = []
    for i, b_type in enumerate(args.types):
        b = generate_bracket(i+1, b_type, reach_probs, ownership, base_bracket, prob_cache, team_to_idx, brackets)
        brackets.append(b)

    # Validate overlap constraint (<85% between any pair)
    MAX_OVERLAP = 0.85
    for i in range(len(brackets)):
        for j in range(i + 1, len(brackets)):
            overlap = compute_bracket_overlap(brackets[i], brackets[j])
            if overlap >= MAX_OVERLAP:
                print(f"  [!] WARNING: Bracket {brackets[i]['bracket_id']} ({brackets[i]['type']}) and "
                      f"Bracket {brackets[j]['bracket_id']} ({brackets[j]['type']}) have {overlap:.0%} overlap (>= {MAX_OVERLAP:.0%} threshold)")

    # Print report
    print()
    for b in brackets:
        print("=" * 50)
        print(f"Bracket {b['bracket_id']} — {b['type'].upper()}")
        print("=" * 50)
        champ = b["champion"]
        c_row = reach_probs[reach_probs["team"] == champ].iloc[0]
        own = ownership.get("champion", {}).get(champ) or get_uniform_ownership("champ")
        print(f"Champion:      {champ} (p_champ: {c_row['p_champ']:.1%}, ownership: {own:.1%}, value: {b['value_score_champion']:.2f})")
        print(f"Final Four:    {' | '.join(b['final_four'])}")
        print(f"Elite Eight:   {' | '.join(b['elite_eight'])}")
        
        if b['upset_picks']:
            print("Upset picks:")
            for u in b['upset_picks']:
                print(f"  -> {u['round'].upper()}: {u['pick']} over opponent in {u['game']} (model: {u['model_prob']:.0%}, public: {u['public_ownership']:.0%})")
        else:
            print("Upset picks:   None")
            
        print(f"Expected score (ESPN standard): {b['expected_score']:.1f}")
        if b['overlap_with_previous'] is not None:
             print(f"Overlap with Prev: {b['overlap_with_previous']:.0%}")
        print("-" * 50)

    with open(out_path, "w") as f:
        json.dump(brackets, f, indent=2)
        
    print(f"\nSaved {len(brackets)} brackets to {out_path}")

if __name__ == "__main__":
    main()
