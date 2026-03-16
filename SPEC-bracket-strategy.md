# SPEC: Bracket Generation Strategy

**Date:** 2026-03-16  
**Status:** Ready for implementation — depends on SPEC-monte-carlo-sim.md completing first  
**Branch:** `jarvis/bracket-strategy` (same branch as sim spec)  

---

## Objective

Build `src/bracket_gen.py` — a module that takes the probability matrix output from the Monte Carlo simulator and generates one or more bracket picks optimized for different pool strategies. Each bracket is a complete set of picks for all 63 tournament games, with a defined champion, Final Four, Elite Eight, Sweet 16, and first/second round picks.

---

## Background: The Strategic Framework

We are targeting two pool contexts simultaneously:

### Pool A — Small Friend Pool
- **Goal:** Maximize expected score. Pure accuracy. Win more often by picking better.
- **Strategy:** Follow model probabilities closely. Differentiate only where model strongly disagrees with conventional wisdom.
- **Bracket count:** 1-2

### Pool B — Large Public Pool (e.g., ESPN Tournament Challenge)
- **Goal:** Beat the field. Expected score alone is insufficient — you need picks the field doesn't have.
- **Strategy:** Pool-aware value picking. Champion and Final Four picks weighted by ownership-adjusted expected value (model probability ÷ public ownership). Intentional differentiation in mid-rounds where public consensus is lopsided and our model disagrees.
- **Bracket count:** Variable (start with 2-4, leave open)

---

## Core Concepts

### 1. Value Score (for Public Pool)
For each team as a potential champion pick:

```
value_score = p_champ / public_ownership_pct
```

A team with `p_champ = 0.20` and `public_ownership = 0.10` has `value_score = 2.0` — you're getting 2x the odds implied by market consensus. A team with `p_champ = 0.30` and `public_ownership = 0.40` has `value_score = 0.75` — negative EV despite being the best team.

**Public ownership source:** ESPN Tournament Challenge champion pick percentages (scraped or manually entered after brackets open Tuesday/Wednesday). Stored as `data/ownership_2026.json`:

```json
{
  "champion": {
    "Duke": 0.28,
    "Michigan": 0.19,
    "Arizona": 0.15,
    "Florida": 0.10,
    ...
  },
  "final_four": {
    "Duke": 0.55,
    "Michigan": 0.44,
    ...
  }
}
```

The script must gracefully handle missing ownership data (default to uniform distribution, warn loudly).

### 2. Round-Level Differentiation Calibrated to Point Value

The key principle: **differentiate where it matters, not randomly.**

Standard ESPN scoring: 1-2-4-8-16-32 per round.

| Round | Points | Differentiation Priority | Trigger for Upset Pick |
|-------|--------|--------------------------|------------------------|
| R64 (32 games) | 1 pt | Low — pick model favorites | Only if p(underdog) ≥ 40% AND public heavily favors top seed |
| R32 (16 games) | 2 pts | Low-Medium | p(underdog) ≥ 38% AND < 30% of public picks them |
| Sweet 16 (8 games) | 4 pts | Medium — worth targeting | p(underdog) ≥ 35% AND < 25% of public picks them |
| Elite Eight (4 games) | 8 pts | High — 1-2 per bracket | p(underdog) ≥ 30% AND < 20% of public picks them |
| Final Four (2 games) | 16 pts | High — 1 per bracket | Value-driven (see champion selection cascade below) |
| Champion (1 pick) | 32 pts | Highest — drives everything | Value score (model prob ÷ ownership) |

**The cascade rule:** The champion pick decision drives backwards. Once you've selected a champion, the bracket must be internally consistent — that team must be picked to win every prior round. Then build outward from there.

### 3. The Upset Pick Threshold

A "model-supported upset" is defined as:
- Our model gives the lower seed ≥ 35% win probability in that specific matchup (not a seed-implied probability — use our model's actual output)
- The public ownership for that lower-seed team winning that round is ≤ 25%

These are the only games where picking the upset earns ownership-adjusted value. Picking upsets elsewhere is just noise.

**Important:** Upset picks must survive bracket logic. Don't pick a 12-seed to reach the Elite Eight if they'd have to beat both a 5-seed and a 1-seed — check that your model gives a plausible path (i.e., cumulative probability of that run is at least 5%).

---

## Bracket Types

### Type 1: Model Chalk
**Use for:** Small friend pool (Pool A), or as the baseline bracket.

Algorithm:
1. For every game, pick the team with higher win probability per our model.
2. Champion = highest `p_champ` team from sim output.
3. No ownership adjustment. Pure expected-score maximization.

### Type 2: Value Champion, Chalk Path
**Use for:** Public pool, moderate differentiation.

Algorithm:
1. Pick champion = highest `value_score` team (model_p_champ / ownership_pct).
2. For Final Four: champion's path is locked. The other three Final Four slots: 2 highest `p_f4` teams from the remaining slots, 1 value pick (highest `p_f4 / ff_ownership` that isn't the champion).
3. Elite Eight through Round 1: follow model probabilities (chalk), *except* apply upset picks where the upset threshold is met.
4. Record which games have upset picks — track the expected ownership differential.

### Type 3: Contrarian
**Use for:** Public pool, maximum differentiation.

Algorithm:
1. Pick champion = second-highest `value_score` team (must differ from Type 2's champion).
2. For Final Four: champion's path locked. At least 2 other Final Four teams must be "non-consensus" (i.e., `p_f4 / ff_ownership > 1.2` and not among the top-3 most commonly picked Final Four teams publicly).
3. Apply upset picks more aggressively: lower the upset threshold from 35% → 30% for Sweet 16 and Elite Eight rounds.
4. Must produce a bracket where the Final Four differs from Type 2 by at least 2 teams.

### Type N: Additional Brackets (Optional)
Additional bracket types beyond 3 are left open. If generating more entries, further champion candidates are drawn down the value_score ranking list (3rd-highest, 4th-highest, etc.). Each new bracket must have a unique champion AND a Final Four that differs from all previous brackets by at least 1 team.

---

## Implementation

### Module: `src/bracket_gen.py`

#### Key Functions

```python
def load_probability_matrix(path: str) -> pd.DataFrame:
    """Load reach_probabilities_2026.csv."""

def load_ownership(path: str) -> dict:
    """Load ownership_2026.json. Warn if file missing; use uniform fallback."""

def compute_value_scores(probs: pd.DataFrame, ownership: dict) -> pd.DataFrame:
    """Add value_score = p_champ / champion_ownership for each team."""

def find_matchup_prob(team_a: str, team_b: str, matchup_cache: dict) -> float:
    """Look up cached per-game win probability (built from predict_matchup)."""

def get_path_to_champion(champion: str, bracket_structure: dict) -> list[str]:
    """
    Return the sequence of opponents the champion must beat, round by round.
    This requires traversing the bracket tree to find who they'd face in each round
    (accounting for which lower bracket half the champion is in).
    Path is probabilistic — pick the most likely opponent at each step.
    """

def pick_round(
    teams_in_slot: list[str],
    matchup_cache: dict,
    upset_threshold: float,
    ownership_round: dict | None,
    ownership_threshold: float,
    locked_winners: dict,
) -> dict[str, str]:
    """
    For each game in a round, pick a winner.
    If a winner is already locked (champion/FF cascade), use it.
    Otherwise: pick model favorite unless upset threshold met.
    Returns {game_slot: winner}.
    """

def generate_bracket(
    bracket_type: str,  # "chalk", "value", "contrarian"
    probs: pd.DataFrame,
    ownership: dict,
    matchup_cache: dict,
    bracket_structure: dict,
    locked_champion: str | None = None,
) -> dict:
    """
    Generate a complete bracket of picks.
    Returns dict with keys: champion, final_four, elite_eight, sweet_16, r32, r64.
    Each value is a list of team names (winners at that round).
    """

def score_bracket(picks: dict, scoring: dict = STANDARD_SCORING) -> float:
    """
    Simulate expected score for a bracket given pick probabilities.
    Expected score = sum over all picks of (prob team actually wins that game * point value).
    """

def compute_bracket_overlap(bracket_a: dict, bracket_b: dict) -> float:
    """Fraction of identical picks between two brackets (0.0 = completely different, 1.0 = identical)."""
```

#### Scoring Constants

```python
STANDARD_SCORING = {
    "r64": 1, "r32": 2, "s16": 4, "e8": 8, "f4": 16, "champ": 32
}
```

---

## CLI Interface

```bash
# Generate brackets with ownership data (standard usage)
python src/bracket_gen.py --ownership data/ownership_2026.json

# Generate specific bracket types
python src/bracket_gen.py --types chalk value contrarian

# Generate N contrarian variants
python src/bracket_gen.py --types chalk value contrarian contrarian contrarian

# Skip ownership (chalk only, or use uniform ownership)
python src/bracket_gen.py --no-ownership

# Save bracket outputs
python src/bracket_gen.py --output data/processed/brackets_2026.json
```

---

## Output Format (`brackets_2026.json`)

```json
[
  {
    "bracket_id": 1,
    "type": "chalk",
    "champion": "Michigan",
    "final_four": ["Michigan", "Duke", "Arizona", "Houston"],
    "elite_eight": ["Michigan", "Iowa St.", "Duke", "UConn", "Arizona", "Purdue", "Florida", "Houston"],
    "sweet_16": [...],
    "r32": [...],
    "r64": [...],
    "expected_score": 84.3,
    "value_score_champion": 1.12,
    "upset_picks": [
      {"round": "s16", "game": "Arizona region", "pick": "Wisconsin", "model_prob": 0.38, "public_ownership": 0.18}
    ],
    "overlap_with_previous": null
  },
  {
    "bracket_id": 2,
    "type": "value",
    ...
    "overlap_with_bracket_1": 0.71
  }
]
```

---

## Report Output (stdout)

For each bracket generated:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bracket 1 — CHALK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Champion:      Michigan (p_champ: 15.1%, ownership: 19.0%, value: 0.79)
Final Four:    Michigan | Duke | Arizona | Houston
Elite Eight:   Michigan | Iowa St. | Duke | UConn | Arizona | Purdue | Florida | Houston
Upset picks:   None
Expected score (ESPN standard): 84.3
────────────────────────────────────────

Bracket 2 — VALUE CHAMPION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Champion:      Florida (p_champ: 12.4%, ownership: 8.2%, value: 1.51)
Final Four:    Florida | Houston | Duke | Iowa St.
Elite Eight:   ...
Upset picks:
  → Sweet 16: Wisconsin over Arkansas (model: 38%, public: 22%)
Expected score (ESPN standard): 79.1
Overlap with Bracket 1: 68%
```

---

## Data Dependency: ownership_2026.json

This file **cannot be generated by code** — it requires manual input from ESPN Tournament Challenge pick percentages (available Tuesday-Wednesday before Thursday's tip-off).

Create `data/ownership_2026.json` as a stub with placeholder values:

```json
{
  "note": "UPDATE WITH REAL OWNERSHIP DATA FROM ESPN/YAHOO BEFORE BRACKETS LOCK",
  "champion": {
    "Duke": 0.25,
    "Michigan": 0.18,
    "Arizona": 0.15,
    "Florida": 0.10,
    "Houston": 0.08,
    "Iowa St.": 0.06,
    "Connecticut": 0.05,
    "Purdue": 0.04
  },
  "final_four": {}
}
```

The bracket_gen script must print a prominent warning if using placeholder ownership data.

---

## Exit Criteria

- [ ] `python src/bracket_gen.py` runs end-to-end without error
- [ ] All generated brackets are internally consistent: champion is picked to win every round they appear in
- [ ] No bracket has the same champion as another bracket in the same run
- [ ] Bracket overlap between any two brackets is < 85% (prevents near-duplicate entries)
- [ ] "Chalk" bracket's champion matches the team with highest `p_champ` in simulation output
- [ ] "Value" bracket's champion differs from "Chalk" bracket's champion (unless one team dominates both metrics)
- [ ] Expected scores are computed and reported for each bracket
- [ ] Upset picks are logged with model probability and public ownership
- [ ] `data/ownership_2026.json` stub created with placeholder values and warning note
- [ ] JSON output saved to `data/processed/brackets_2026.json`

---

## What NOT to Do

- ❌ Do not generate brackets with random upset picks unrelated to model probabilities — every non-chalk pick must be supported by a model probability above the threshold
- ❌ Do not hardcode specific team names — everything must flow from `reach_probabilities_2026.csv`
- ❌ Do not try to maximize "diversity" for its own sake — differentiation is only valuable where ownership data supports it
- ❌ Do not force a set number of upset picks per bracket — the number of valid upsets is data-driven
- ❌ Do not make champion selection based on seed alone — model probability and value score are the only inputs
- ❌ Do not implement "path diversity" or try to generate maximally different brackets — that was the scrapped approach. Brackets differ because they have different champions and cascade accordingly.

---

## Negative Space (Approaches Considered and Rejected)

- **Monte Carlo bracket generation (diverse sampling):** Was the original approach. Produced many brackets but poor signal — diversity for its own sake doesn't improve EV. Scrapped.
- **Temperature-based sampling:** Used in the old `bracket_engine.py`. Generates diverse picks but not ownership-calibrated. Scrapped.
- **Fixed number of brackets (3):** We discussed 3 initially, but left the count open. The right number depends on pool count and available distinct value picks. Don't hardcode 3.
- **Upset picks based on historical seed trends (e.g., "always pick a 12 over 5"):** Explicitly rejected. Use model probabilities, not historical base rates.
