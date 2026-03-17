"""
Bracket Generation Module v1.5

Probabilistic bracket generation with temperature-controlled upset flipping
and greedy portfolio optimization for E[max score].
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data_prep import load_team_stats
from predict import load_models, predict_matchup
from simulate import (
    load_bracket, extract_teams_from_bracket, build_probability_cache,
    simulate_bracket_raw, get_regions_order,
)

STANDARD_SCORING = {"r64": 1, "r32": 2, "s16": 4, "e8": 8, "f4": 16, "champ": 32}

# Seed -> matchup position within a region (index into config.MATCHUP_ORDER)
_SEED_TO_POS = {}
for _pos, (_s1, _s2) in enumerate(config.MATCHUP_ORDER):
    _SEED_TO_POS[_s1] = _pos
    _SEED_TO_POS[_s2] = _pos


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
        if "UPDATE WITH REAL OWNERSHIP DATA" in str(data.get("note", "")):
            print("  [!] WARNING: Using placeholder ownership data from JSON note.")
        return data


def resolve_first_four(tbd_str: str, prob_cache: np.ndarray, team_to_idx: dict) -> str:
    """Resolve a TBD_TeamA_TeamB string by picking the model favorite."""
    parts = tbd_str[4:].split("_")
    if len(parts) != 2:
        print(f"  [!] Could not parse First Four entry '{tbd_str}', using '{parts[0]}'")
        return parts[0]
    t1, t2 = parts[0], parts[1]
    if t1 not in team_to_idx or t2 not in team_to_idx:
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
    matchups = []
    for s1, s2 in config.MATCHUP_ORDER:
        t1, t2 = region_seeds[str(s1)], region_seeds[str(s2)]
        if t1.startswith("TBD_"): t1 = resolve_first_four(t1, prob_cache, team_to_idx)
        if t2.startswith("TBD_"): t2 = resolve_first_four(t2, prob_cache, team_to_idx)
        matchups.append((t1, t2))
    return matchups


def get_p_win(t1: str, t2: str, prob_cache: np.ndarray, team_to_idx: dict) -> float:
    """Get probability that t1 beats t2."""
    return prob_cache[team_to_idx[t1], team_to_idx[t2]]


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


# ---------------------------------------------------------------------------
# v1.5 -- Probabilistic Portfolio Generation
# ---------------------------------------------------------------------------

def build_champion_pool(reach_probs: pd.DataFrame, cutoff: float = None) -> pd.DataFrame:
    """Build champion candidate pool from top teams by p_champ until cumulative >= cutoff."""
    if cutoff is None:
        cutoff = config.BRACKET_CHAMP_CUMULATIVE_CUTOFF

    sorted_df = reach_probs.sort_values("p_champ", ascending=False).reset_index(drop=True)
    cumsum = sorted_df["p_champ"].cumsum()

    # Include teams until cumulative >= cutoff
    mask = cumsum >= cutoff
    if mask.any():
        n_include = int(mask.idxmax()) + 1
    else:
        n_include = len(sorted_df)

    pool = sorted_df.iloc[:n_include].copy()
    pool["weight"] = pool["p_champ"] / pool["p_champ"].sum()

    print(f"  Champion pool: {len(pool)} teams covering {pool['p_champ'].sum():.1%} of championship mass")
    for _, row in pool.iterrows():
        print(f"    {row['team']:<20} p_champ={row['p_champ']:.1%}  weight={row['weight']:.1%}")

    return pool


def get_champion_locked_slots(
    champion: str,
    reach_probs: pd.DataFrame,
    base_bracket: dict,
    regions_order: list[str],
) -> set[int]:
    """Compute the 6 game slot indices where champion must win (R64 through Championship)."""
    champ_row = reach_probs[reach_probs["team"] == champion].iloc[0]
    region = champ_row["region"]
    seed = int(champ_row["seed"])

    region_idx = regions_order.index(region)
    matchup_pos = _SEED_TO_POS[seed]

    locked = set()
    locked.add(region_idx * 8 + matchup_pos)                    # R64
    locked.add(32 + region_idx * 4 + matchup_pos // 2)          # R32
    locked.add(48 + region_idx * 2 + matchup_pos // 4)          # S16
    locked.add(56 + region_idx)                                  # E8

    # F4: find which pair the champion's region belongs to
    for pair_idx, pair in enumerate(base_bracket["final_four_matchups"]):
        if region in pair:
            locked.add(60 + pair_idx)
            break

    locked.add(62)                                               # Championship

    return locked


def _resolve_game(
    a_idx: int, b_idx: int,
    temperature: float,
    prob_cache: np.ndarray,
    idx_to_team: dict,
    ownership: dict,
    round_name: str,
    p_floor: float,
    rng: np.random.Generator,
) -> tuple[int, bool]:
    """Resolve a single game probabilistically. Returns (winner_idx, is_upset)."""
    p_a = float(prob_cache[a_idx, b_idx])

    if p_a >= 0.5:
        fav_idx, dog_idx = a_idx, b_idx
        p_underdog = 1.0 - p_a
    else:
        fav_idx, dog_idx = b_idx, a_idx
        p_underdog = p_a

    if p_underdog < p_floor:
        return fav_idx, False

    # Compute upset score: model probability * contrarian opportunity
    dog_name = idx_to_team[dog_idx]
    dog_own = ownership.get(round_name, {}).get(dog_name)
    if dog_own is None:
        dog_own = get_uniform_ownership(round_name)

    upset_score = p_underdog * (1.0 - dog_own)

    # Temperature-controlled flip: upset_score^(1/temperature)
    p_flip = upset_score ** (1.0 / temperature)

    if rng.random() < p_flip:
        return dog_idx, True
    return fav_idx, False


def generate_bracket_probabilistic(
    champion_idx: int,
    temperature: float,
    r64_team_pairs: list[tuple[int, int]],
    prob_cache: np.ndarray,
    idx_to_team: dict,
    ownership: dict,
    p_floor: float,
    locked_slots: set[int],
    rng: np.random.Generator,
) -> dict:
    """Generate one bracket probabilistically with locked champion path.

    Args:
        r64_team_pairs: list of 32 (idx_a, idx_b) tuples in standard slot order
        locked_slots: set of slot indices where champion must win
    """
    picks_idx = np.zeros(63, dtype=np.int16)
    upset_count = 0

    # R64: 32 games (slots 0-31)
    for slot in range(32):
        a_idx, b_idx = r64_team_pairs[slot]
        if slot in locked_slots:
            picks_idx[slot] = champion_idx
            continue

        winner, is_upset = _resolve_game(
            a_idx, b_idx, temperature, prob_cache, idx_to_team,
            ownership, "r64", p_floor, rng
        )
        picks_idx[slot] = winner
        if is_upset:
            upset_count += 1

    # R32 through Championship
    round_defs = [
        (32, 16, "r32"),
        (48, 8, "s16"),
        (56, 4, "e8"),
        (60, 2, "f4"),
        (62, 1, "champ"),
    ]
    prev_start = 0
    for round_start, n_games, round_name in round_defs:
        for g in range(n_games):
            slot = round_start + g

            # Get the two teams from previous round winners
            a_idx = int(picks_idx[prev_start + 2 * g])
            b_idx = int(picks_idx[prev_start + 2 * g + 1])

            if slot in locked_slots:
                picks_idx[slot] = champion_idx
                continue

            winner, is_upset = _resolve_game(
                a_idx, b_idx, temperature, prob_cache, idx_to_team,
                ownership, round_name, p_floor, rng
            )
            picks_idx[slot] = winner
            if is_upset:
                upset_count += 1

        prev_start = round_start

    return {
        "picks_idx": picks_idx,
        "champion": idx_to_team[int(picks_idx[62])],
        "temperature": temperature,
        "upset_count": upset_count,
    }


def generate_all_brackets(
    n_total: int,
    champion_pool: pd.DataFrame,
    temp_tiers: list[tuple[float, float, float]],
    r64_team_pairs: list[tuple[int, int]],
    prob_cache: np.ndarray,
    team_to_idx: dict,
    idx_to_team: dict,
    ownership: dict,
    p_floor: float,
    reach_probs: pd.DataFrame,
    base_bracket: dict,
    regions_order: list[str],
) -> tuple[np.ndarray, list[dict]]:
    """Generate n_total candidate brackets with temperature-stratified sampling.

    Returns:
        bracket_picks: (n_total, 63) int16 array of team indices per slot
        bracket_meta: list of dicts with champion, temperature, upset_count
    """
    # Pre-assign temperatures from tier distribution
    rng_master = np.random.default_rng(42)
    temperatures = []
    for lo, hi, frac in temp_tiers:
        n_tier = int(n_total * frac)
        temperatures.extend(rng_master.uniform(lo, hi, n_tier).tolist())
    while len(temperatures) < n_total:
        temperatures.append(rng_master.uniform(temp_tiers[1][0], temp_tiers[1][1]))
    temperatures = temperatures[:n_total]
    rng_master.shuffle(temperatures)

    # Champion sampling setup
    champ_names = champion_pool["team"].values
    champ_weights = champion_pool["weight"].values
    champ_indices = np.array([team_to_idx[t] for t in champ_names])

    # Precompute locked slots for each champion in the pool
    champ_locked_cache = {}
    for t in champ_names:
        champ_locked_cache[t] = get_champion_locked_slots(
            t, reach_probs, base_bracket, regions_order
        )

    bracket_picks = np.zeros((n_total, 63), dtype=np.int16)
    bracket_meta = []

    print(f"\nGenerating {n_total:,} candidate brackets...")
    start_time = time.time()

    for i in range(n_total):
        rng = np.random.default_rng(seed=i)

        # Sample champion from pool
        champ_pool_idx = rng.choice(len(champ_names), p=champ_weights)
        champ_name = champ_names[champ_pool_idx]
        champ_team_idx = int(champ_indices[champ_pool_idx])
        locked = champ_locked_cache[champ_name]

        result = generate_bracket_probabilistic(
            champion_idx=champ_team_idx,
            temperature=temperatures[i],
            r64_team_pairs=r64_team_pairs,
            prob_cache=prob_cache,
            idx_to_team=idx_to_team,
            ownership=ownership,
            p_floor=p_floor,
            locked_slots=locked,
            rng=rng,
        )

        bracket_picks[i] = result["picks_idx"]
        bracket_meta.append({
            "champion": result["champion"],
            "temperature": result["temperature"],
            "upset_count": result["upset_count"],
        })

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            print(f"  {i+1:,}/{n_total:,} brackets generated ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"  All {n_total:,} brackets generated in {elapsed:.1f}s")

    # Print champion distribution
    champ_counts = {}
    for m in bracket_meta:
        champ_counts[m["champion"]] = champ_counts.get(m["champion"], 0) + 1
    print("  Champion distribution:")
    for t, c in sorted(champ_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<20} {c:>5} ({c/n_total:.1%})")

    # Print temperature vs upset stats
    low_upsets = [m["upset_count"] for m in bracket_meta if m["temperature"] < 0.4]
    high_upsets = [m["upset_count"] for m in bracket_meta if m["temperature"] > 1.5]
    if low_upsets and high_upsets:
        print(f"  Avg upsets: low-temp={np.mean(low_upsets):.1f}, high-temp={np.mean(high_upsets):.1f}")

    return bracket_picks, bracket_meta


def select_portfolio_greedy(
    bracket_picks: np.ndarray,
    sim_outcomes: np.ndarray,
    round_points: np.ndarray,
    n_portfolio: int,
    chunk_size: int = None,
) -> list[dict]:
    """Select portfolio via greedy E[max score] optimization.

    Precomputes all bracket scores against simulations (uint8 matrix),
    then runs greedy selection over the precomputed matrix.

    Returns list of dicts with index, e_max_after, marginal_gain.
    """
    if chunk_size is None:
        chunk_size = config.BRACKET_SCORE_CHUNK_SIZE

    n_brackets = bracket_picks.shape[0]
    n_sims = sim_outcomes.shape[0]

    print(f"\nSelecting portfolio of {n_portfolio} from {n_brackets:,} candidates "
          f"against {n_sims:,} simulations...")
    start_time = time.time()

    # Precompute score matrix: (n_brackets, n_sims) uint8
    # Per-bracket dot product with preallocated buffer to avoid allocation overhead
    print(f"  Precomputing score matrix ({n_brackets:,} x {n_sims:,})...")
    score_start = time.time()
    all_scores = np.zeros((n_brackets, n_sims), dtype=np.uint8)
    rp32 = round_points.astype(np.float32)
    matches_buf = np.empty((n_sims, 63), dtype=np.float32)
    sim_f32 = sim_outcomes.astype(np.float32)
    for i in range(n_brackets):
        np.equal(bracket_picks[i].astype(np.float32), sim_f32, out=matches_buf)
        all_scores[i] = (matches_buf @ rp32).astype(np.uint8)
        if (i + 1) % 2000 == 0:
            elapsed_so_far = time.time() - score_start
            print(f"    {i+1:,}/{n_brackets:,} scored ({elapsed_so_far:.1f}s)...")
    score_elapsed = time.time() - score_start
    print(f"  Score matrix complete in {score_elapsed:.1f}s "
          f"({all_scores.nbytes / 1e6:.0f} MB)")

    # Greedy selection over precomputed scores
    selected = []
    selected_indices = set()
    current_max_scores = None
    prev_expected = 0.0

    for iteration in range(n_portfolio):
        best_idx = -1
        best_expected = -1.0

        iter_start = time.time()

        # Process in chunks to control memory (chunk x K float32 temporaries)
        for start in range(0, n_brackets, chunk_size):
            end = min(start + chunk_size, n_brackets)
            chunk = all_scores[start:end].astype(np.float32)  # (c, K)

            if current_max_scores is None:
                means = chunk.mean(axis=1)
            else:
                combined = np.maximum(chunk, current_max_scores[np.newaxis, :])
                means = combined.mean(axis=1)

            # Mask already-selected brackets
            for sel_idx in selected_indices:
                if start <= sel_idx < end:
                    means[sel_idx - start] = -1.0

            local_best = int(means.argmax())
            if means[local_best] > best_expected:
                best_expected = float(means[local_best])
                best_idx = start + local_best

        marginal = best_expected - prev_expected
        selected.append({
            "index": best_idx,
            "e_max_after": best_expected,
            "marginal_gain": marginal,
        })
        selected_indices.add(best_idx)

        if current_max_scores is None:
            current_max_scores = all_scores[best_idx].astype(np.float32).copy()
        else:
            current_max_scores = np.maximum(
                current_max_scores, all_scores[best_idx].astype(np.float32)
            )

        prev_expected = best_expected

        iter_elapsed = time.time() - iter_start
        print(f"  Portfolio bracket {iteration + 1}: idx={best_idx}, "
              f"E[max]={best_expected:.1f}, marginal=+{marginal:.1f} ({iter_elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"  Portfolio selection complete in {elapsed:.1f}s")

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bracket Generator v1.5 -- Probabilistic Portfolio")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--ownership", type=str, default="")
    parser.add_argument("--output", type=str,
                        default=str(config.PROCESSED_DIR / "brackets_{}.json"))
    parser.add_argument("--n-total", type=int, default=config.BRACKET_N_TOTAL)
    parser.add_argument("--n-sims", type=int, default=config.BRACKET_N_SIMS)
    parser.add_argument("--n-portfolio", type=int, default=config.BRACKET_N_PORTFOLIO)
    args = parser.parse_args()

    out_path = Path(args.output.replace("{}", str(args.season)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load models and data ---
    print("Loading models and data...")
    models, scaler, calibrators = load_models()
    stats = load_team_stats()

    bracket_path = config.PROJECT_ROOT / "data" / f"bracket_{args.season}.json"
    if not bracket_path.exists():
        print(f"  [X] Bracket file {bracket_path} not found.")
        sys.exit(1)

    base_bracket = load_bracket(bracket_path)
    team_names = extract_teams_from_bracket(base_bracket)
    prob_cache, team_to_idx = build_probability_cache(
        team_names, stats, args.season, models, scaler, config.FEATURES, calibrators
    )
    idx_to_team = {v: k for k, v in team_to_idx.items()}

    # Load reach probabilities
    prob_csv_path = config.PROCESSED_DIR / f"reach_probabilities_{args.season}.csv"
    if not prob_csv_path.exists():
        print(f"  [X] Reach probabilities {prob_csv_path} not found. Run simulate.py first.")
        sys.exit(1)
    reach_probs = pd.read_csv(prob_csv_path)

    # Load ownership
    ownership = {}
    if args.ownership:
        ownership = load_ownership(args.ownership)

    regions_order = get_regions_order(base_bracket)

    # --- Build R64 team pairs (resolved, deterministic) ---
    print("\nResolving First Four and building R64 matchups...")
    r64_team_pairs = []
    for region in regions_order:
        matchups = build_region_round1(
            base_bracket["regions"][region], prob_cache, team_to_idx
        )
        for t1, t2 in matchups:
            r64_team_pairs.append((team_to_idx[t1], team_to_idx[t2]))

    # --- Stage 1: Champion Pool ---
    print("\n--- Stage 1: Champion Pool ---")
    champion_pool = build_champion_pool(reach_probs)

    # --- Stage 2: Generate Brackets ---
    print("\n--- Stage 2: Bracket Generation ---")
    bracket_picks, bracket_meta = generate_all_brackets(
        n_total=args.n_total,
        champion_pool=champion_pool,
        temp_tiers=config.BRACKET_TEMP_TIERS,
        r64_team_pairs=r64_team_pairs,
        prob_cache=prob_cache,
        team_to_idx=team_to_idx,
        idx_to_team=idx_to_team,
        ownership=ownership,
        p_floor=config.BRACKET_P_FLOOR,
        reach_probs=reach_probs,
        base_bracket=base_bracket,
        regions_order=regions_order,
    )

    # --- Stage 3: Tournament Simulation ---
    print("\n--- Stage 3: Tournament Simulation ---")
    sim_outcomes, round_points = simulate_bracket_raw(
        args.n_sims, base_bracket, prob_cache, team_to_idx
    )

    # --- Stage 4: Portfolio Selection ---
    print("\n--- Stage 4: Portfolio Selection ---")
    portfolio = select_portfolio_greedy(
        bracket_picks, sim_outcomes, round_points, args.n_portfolio
    )

    # --- Build output ---
    print("\n--- Results ---")
    output = {
        "metadata": {
            "n_candidates": args.n_total,
            "n_sims": args.n_sims,
            "n_portfolio": args.n_portfolio,
            "champion_pool_size": len(champion_pool),
            "p_floor": config.BRACKET_P_FLOOR,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "portfolio": [],
    }

    portfolio_dicts = []
    for port_idx, sel in enumerate(portfolio):
        idx = sel["index"]
        meta = bracket_meta[idx]
        picks = bracket_picks[idx]

        pick_names = [idx_to_team[int(p)] for p in picks]

        bracket_dict = {
            "bracket_id": port_idx + 1,
            "champion": pick_names[62],
            "temperature": round(meta["temperature"], 3),
            "final_four": [pick_names[i] for i in range(56, 60)],
            "elite_eight": [pick_names[i] for i in range(56, 60)],
            "sweet_16": [pick_names[i] for i in range(48, 56)],
            "r32": [pick_names[i] for i in range(32, 48)],
            "r64": [pick_names[i] for i in range(0, 32)],
            "upset_count": meta["upset_count"],
            "e_max_score_after": round(sel["e_max_after"], 1),
            "marginal_gain": round(sel["marginal_gain"], 1),
        }
        portfolio_dicts.append(bracket_dict)
        output["portfolio"].append(bracket_dict)

    # Compute pairwise overlaps
    for i, bd in enumerate(portfolio_dicts):
        overlaps = []
        for j, other in enumerate(portfolio_dicts):
            if i != j:
                overlaps.append(round(compute_bracket_overlap(bd, other), 2))
        bd["overlap_with_others"] = overlaps

    # Print report
    for bd in portfolio_dicts:
        print()
        print("=" * 60)
        print(f"Bracket {bd['bracket_id']} (temp={bd['temperature']:.3f}, upsets={bd['upset_count']})")
        print("=" * 60)
        print(f"  Champion:    {bd['champion']}")
        print(f"  Final Four:  {' | '.join(bd['final_four'])}")
        print(f"  Elite Eight: {' | '.join(bd['sweet_16'])}")
        print(f"  E[max]:      {bd['e_max_score_after']}")
        print(f"  Marginal:    +{bd['marginal_gain']}")
        print(f"  Overlap:     {bd['overlap_with_others']}")
        print("-" * 60)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(portfolio_dicts)} portfolio brackets to {out_path}")


if __name__ == "__main__":
    main()
