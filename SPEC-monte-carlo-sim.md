# SPEC: Monte Carlo Tournament Simulator

**Date:** 2026-03-16  
**Status:** Ready for implementation  
**Branch:** Create `jarvis/bracket-strategy` from main  

---

## Objective

Build `src/simulate.py` — a Monte Carlo simulator that takes the 2026 bracket structure (region assignments, seeds, first-round matchups) and propagates our model's calibrated per-game win probabilities through all 6 rounds of the tournament. Output: per-team probability of reaching each round (including championship), plus expected bracket score under a configurable scoring system.

---

## Context

Our model (`models/trained_models.pkl`) produces calibrated ensemble win probabilities for any head-to-head matchup given BartTorvik season stats. The existing `src/predict.py` predicts individual games in isolation. What we need is path-aware simulation: a team that wins round 1 faces the winner of another game in round 2, so round 2+ probabilities are conditional on who survived earlier rounds.

The existing `src/simulate.py` and `src/bracket_engine.py` on `main` exist but were built for a scrapped diversity-based bracket generation approach. **Do not reuse or extend them.** Build fresh in `src/simulate.py` (overwrite).

---

## Inputs

### 1. Bracket Structure (`data/bracket_2026.json`)
Create this file as part of implementation. Structure:

```json
{
  "regions": {
    "East": {
      "1": "Duke", "16": "Siena",
      "8": "Ohio St.", "9": "TCU",
      "5": "St. John's", "12": "Northern Iowa",
      "4": "Kansas", "13": "Cal Baptist",
      "6": "Louisville", "11": "South Florida",
      "3": "Michigan St.", "14": "North Dakota St.",
      "7": "UCLA", "10": "UCF",
      "2": "Connecticut", "15": "Furman"
    },
    "Midwest": {
      "1": "Michigan", "16": "TBD_UMBC_Howard",
      "8": "Georgia", "9": "Saint Louis",
      "5": "Texas Tech", "12": "Akron",
      "4": "Alabama", "13": "Hofstra",
      "6": "Tennessee", "11": "TBD_MiamiOH_SMU",
      "3": "Virginia", "14": "Wright St.",
      "7": "Kentucky", "10": "Santa Clara",
      "2": "Iowa St.", "15": "Tennessee St."
    },
    "South": {
      "1": "Florida", "16": "TBD_PV_Lehigh",
      "8": "Clemson", "9": "Iowa",
      "5": "Vanderbilt", "12": "McNeese St.",
      "4": "Nebraska", "13": "Troy",
      "6": "North Carolina", "11": "VCU",
      "3": "Illinois", "14": "Penn",
      "7": "Saint Mary's", "10": "Texas A&M",
      "2": "Houston", "15": "Idaho"
    },
    "West": {
      "1": "Arizona", "16": "LIU",
      "8": "Villanova", "9": "Utah St.",
      "5": "Wisconsin", "12": "High Point",
      "4": "Arkansas", "13": "Hawaii",
      "6": "BYU", "11": "TBD_TX_NCST",
      "3": "Gonzaga", "14": "Kennesaw St.",
      "7": "Miami FL", "10": "Missouri",
      "2": "Purdue", "15": "Queens"
    }
  },
  "final_four_matchups": [
    ["East", "West"],
    ["Midwest", "South"]
  ]
}
```

**First Four handling:** Four slots marked `TBD_*`. Before running the main simulation, the simulator resolves these by either (a) using our model's win probability for the First Four game to pick the more probable team, or (b) running the sim twice for each outcome and averaging. Default: pick the higher-probability team (simpler, sufficient). Make this configurable via `--first-four-mode {favorite|average}`.

### 2. Team Stats
Load from `team_stats.csv` via `src/data_prep.load_team_stats()` filtered to `Season == 2026`. Already populated with seeds.

### 3. Trained Models
Load from `models/trained_models.pkl` via `src/predict.load_models()`.

---

## Algorithm

### Per-Game Win Probability
Use `src/predict.predict_matchup()` directly. It takes two team stat rows and returns calibrated ensemble win probability for team A. Reuse without modification.

**Important:** Cache all pairwise probabilities at startup. With 60 teams, there are at most 60×59/2 = 1770 unique matchups. Computing each once upfront (rather than per-simulation-step) is fast and avoids redundant lookups during Monte Carlo runs.

### Bracket Simulation (Single Run)
The standard NCAA bracket structure is a fixed single-elimination tree. The seeding within each region follows this pairing:

```
Round 1 pods (within each region, 4 games per pod):
  Pod A: 1 vs 16, 8 vs 9
  Pod B: 5 vs 12, 4 vs 13
  Pod C: 6 vs 11, 3 vs 14
  Pod D: 7 vs 10, 2 vs 15

Round 2 (Sweet 16 path):
  Pod A winner vs Pod B winner → Regional Semifinal slot 1
  Pod C winner vs Pod D winner → Regional Semifinal slot 2

Round 3 (Elite Eight): Regional Semi 1 vs Regional Semi 2 → Regional winner
Round 4 (Final Four): East/West regional winners meet; Midwest/South regional winners meet
Round 5 (Championship): Final Four winners meet
```

For each simulation run:
1. For each game, draw a Bernoulli(p) where p = cached win probability for team A.
2. Winner advances. Repeat until one team remains.
3. Record which team reached each round (R64 through Champion).

### Monte Carlo
- Default: **50,000 simulations** (configurable via `--n-sims`)
- Parallelizable via `numpy` vectorized sampling — avoid Python loops per simulation where possible.
- Efficient implementation: represent the bracket as a fixed tree of 63 game slots. Pre-fill probabilities from cache. Run all N simulations in vectorized form using `numpy.random.random((N, 63))` compared against probability thresholds.

### Output: Probability Matrix

For each team, compute:
- `p_r64`: probability of winning round 1 (first round)
- `p_r32`: probability of reaching Sweet 16 (winning 2 games)
- `p_s16`: probability of reaching Elite 8
- `p_e8`: probability of reaching Final Four
- `p_f4`: probability of reaching Championship game
- `p_champ`: probability of winning championship
- `expected_score_standard`: expected bracket points under standard ESPN scoring (1-2-4-8-16-32), assuming you pick that team to win every round they appear in
- `expected_score_seeded`: expected bracket points under seed-differential bonus scoring (for reference)

Save to `data/processed/reach_probabilities_2026.csv`.

Print a sorted table (by `p_champ` descending) to stdout.

---

## Expected Score Calculation

For the standard ESPN scoring system (1-2-4-8-16-32 per round), the expected points from picking team X to win the championship is:

```
E[score | pick X as champion] = 
    p_r64 * 1 + p_r32 * 2 + p_s16 * 4 + p_e8 * 8 + p_f4 * 16 + p_champ * 32
```

This is the maximum expected score you can get from team X's slot — it assumes you also pick them to win every prior round. This is the right quantity for comparing champion candidates.

---

## CLI Interface

```bash
# Full simulation with default settings
python src/simulate.py

# Custom number of simulations
python src/simulate.py --n-sims 100000

# Specific season (default: latest with seeds)
python src/simulate.py --season 2026

# First Four mode
python src/simulate.py --first-four-mode average

# Output path override
python src/simulate.py --output data/processed/my_probs.csv
```

---

## Output Format (`reach_probabilities_2026.csv`)

| Column | Type | Description |
|--------|------|-------------|
| team | str | BartTorvik team name |
| seed | int | Tournament seed (1-16) |
| region | str | East/West/South/Midwest |
| p_r64 | float | P(win round 1) |
| p_r32 | float | P(reach Sweet 16) |
| p_s16 | float | P(reach Elite Eight) |
| p_e8 | float | P(reach Final Four) |
| p_f4 | float | P(reach Championship) |
| p_champ | float | P(win championship) |
| expected_score | float | Expected ESPN points if picked as champion through all rounds |

---

## Stdout Report (example format)

```
2026 NCAA Tournament Probability Matrix (50,000 simulations)
══════════════════════════════════════════════════════════════════════════════
Team              Seed Region    R1     R32    S16    E8     F4     Champ  E[Pts]
──────────────────────────────────────────────────────────────────────────────
Duke              1    East    93.1%  72.4%  51.3%  35.2%  22.1%  14.3%   43.2
Michigan          1    Midwest 92.2%  71.8%  53.1%  37.4%  23.8%  15.1%   45.6
...
```

---

## Exit Criteria

- [ ] `python src/simulate.py` runs without error on the 2026 bracket
- [ ] Output CSV contains all 60 tournament teams (+ First Four replacements)
- [ ] Championship probabilities sum to ~1.0 (within 0.5% of 1.0)
- [ ] P(reach each round) is monotonically non-increasing as rounds advance for every team
- [ ] At least one 1-seed has `p_champ` > 10%
- [ ] At least one 12-seed or higher has `p_champ` < 1%
- [ ] Expected score values are all positive and correlate with `p_champ`
- [ ] `data/bracket_2026.json` created and committed
- [ ] `data/processed/reach_probabilities_2026.csv` saved

---

## What NOT to Do

- ❌ Do not reuse `src/bracket_engine.py` or the old `src/simulate.py` logic — they were built for diversity generation, not probability estimation
- ❌ Do not simulate path diversity or try to generate bracket picks here — that's a separate concern (see SPEC-bracket-strategy.md)
- ❌ Do not retrain models — use `models/trained_models.pkl` as-is
- ❌ Do not add First Four teams as named entities in BartTorvik stats if they aren't already there — use the higher-probability team from each First Four game as the slot's representative
