# Spec: Bracket Diversity — Upset-Aware Simulation

## Diagnosis

The model itself is NOT broken. First-round win probabilities are reasonable:
- Colorado State (12) has 66% to beat Memphis (5) ← model got this right
- McNeese (12) has 34% to beat Clemson (5) ← reasonable for a 12-seed
- Arkansas (10) has 44% vs St. John's (2) ← nearly a coin flip, appropriate
- Vanderbilt (10) has 27% vs Saint Mary's (7) ← correctly an underdog

The simulation also correctly computes reach probabilities through Monte Carlo.

**The bug is in the bracket generator, not the model or sim.**

`generate_greedy_bracket()` picks the highest-EV team at every matchup, which
is almost always the lower-seeded team. The result: every bracket looks like a
1-seed parade. The variance that exists in the sim is discarded at bracket-generation time.

**Root causes:**
1. `pick_winner_by_ev()` multiplies P(win) × round_points — this always favors
   the team more likely to *reach* that round, which compounds chalk picks
2. No bracket generation strategy actually samples from the sim's distribution —
   they all deterministically pick the mode
3. The upset special and wildcard strategies try to force specific teams deep,
   but the forced team usually has low championship probability, making the
   resulting bracket incoherent (a 10-seed champion with a chalk Final Four)

---

## Goal

Generate bracket variants that are:
1. **Internally consistent** — if you pick an upset, follow that team's path
   through subsequent rounds logically (their next opponent changes)
2. **Probability-weighted** — sample from the sim distribution rather than
   always picking the mode
3. **Strategically diverse** — produce brackets that genuinely differ in
   late-round picks, not just in the champion slot

---

## Context

Relevant files:
- `src/simulate.py` — all bracket generation logic lives here
- `data/processed/reach_probabilities_2025.csv` — sim output to validate against
- `data/bracket_2025.json` — bracket structure input

Key functions to modify or replace:
- `generate_greedy_bracket()` — needs probabilistic sampling variant
- `generate_all_brackets()` — needs new strategy implementations
- `get_upset_candidates_for_bracket()` — needs to surface regionally-coherent candidates

---

## Tasks

- [ ] 1. **Add `generate_sampled_bracket()` function**

  Instead of always picking the highest-EV team, sample from the probability
  distribution with optional temperature control:
  - `temperature=1.0` → pure probability sampling (sample sim results directly)
  - `temperature=0.0` → fully greedy (current behavior, keep as Bracket 1)
  - `temperature=0.5` → blend: softens probabilities toward 50/50 but still
    respects the model's signal

  Crucially: **simulate the entire bracket path together.** When a team wins
  round 1, their round 2 opponent is whoever won the adjacent matchup — not
  a fixed team. This means the bracket must be simulated game-by-game in
  bracket order, not independently.

  Pseudocode:
  ```python
  def generate_sampled_bracket(region_matchups, ff_matchups, cache, temperature=0.5):
      # For each region, simulate round by round in bracket order
      # At each matchup: p_adj = soften(p, temperature)
      # Sample winner: random() < p_adj → home wins
      # Advance winner to face the winner of the adjacent bracket slot
  ```

- [ ] 2. **Fix bracket order / adjacency**

  Current code pairs `round_winners[i]` with `round_winners[i+1]`, which is
  correct for bracket structure. Verify this is preserved in the new sampled
  generator — the bracket tree must be maintained so that upset teams face
  the correct subsequent opponents.

- [ ] 3. **Replace upset special strategy with "coherent upset path"**

  Current approach: force a specific team to win their region, then run greedy
  everywhere else. Problem: the forced team's early opponents are still chosen
  greedily, producing an inconsistent bracket.

  New approach:
  - Identify top upset candidates by region using `get_upset_candidates_for_bracket()`
  - For the "upset special" bracket: use `temperature=0.7` (higher variance)
    seeded with a random state that's likely to produce an underdog run
  - OR: force the upset team's path explicitly — if Colorado State (12) is the
    upset pick, manually set them to beat Memphis (5) in round 1, then run
    sampled from there

- [ ] 4. **Update `generate_all_brackets()` to use new strategies**

  Replace the 5 bracket strategies with:
  - Bracket 1: `temperature=0.0` — pure greedy EV (chalk, highest floor)
  - Bracket 2: `temperature=0.0`, champion = #2 probability team (alt chalk)
  - Bracket 3: `temperature=0.5` — balanced sampling (most likely to score well)
  - Bracket 4: `temperature=0.8`, seeded for max variance — "chaos bracket"
  - Bracket 5: Coherent upset path — pick top upset candidate per region,
    force their round-1 win, sample the rest at temperature=0.5

- [ ] 5. **Add bracket diversity metric to output**

  After generating all N brackets, compute and print a pairwise overlap score:
  ```
  overlap(A, B) = picks in common / total picks
  ```
  Target: Brackets 1 and 5 should have < 60% overlap. If all brackets are
  >80% overlap, warn the user and suggest raising temperature.

- [ ] 6. **Validate on 2025 data**

  Run with `--season 2025 --n-sims 10000 --n-brackets 5` and confirm:
  - At least one bracket has a non-1-seed champion
  - At least one of the 2025 actual upsets (Colorado State, McNeese, Drake,
    Arkansas) appears in at least one bracket
  - Brackets 1 and 5 have <70% pick overlap
  - Expected scores are still reasonable (not cratering below 80)

---

## Constraints

- DO: Preserve `temperature=0.0` (greedy) as Bracket 1 — it's the floor
- DO: Keep bracket adjacency correct — upsets must face the correct next opponent
- DO: Use the logistic model's probability (best single model) as the base
- DON'T: Change the Monte Carlo simulation itself — it's correct
- DON'T: Change `predict_matchup()` or any model code
- DON'T: Modify the bracket JSON format or loading logic
- Branch: `feature/bracket-simulation` (already exists, continue there)

## Done When

`python3 src/simulate.py --season 2025 --n-sims 10000 --n-brackets 5` produces:
1. A table of 5 brackets with visibly different champion/Final Four picks
2. A diversity report showing pairwise overlap scores
3. At least one bracket containing a known 2025 upset team in the Sweet 16+
4. No crashes, no degradation to reach probability computation

## Rules & Tips

- The Monte Carlo sim is correct and untouched — it already captures variance.
  The problem is purely in how brackets are generated *from* that sim output.
- Temperature=0.5 means: if P(A)=0.7, adjust to P_adj = softmax([0.7, 0.3] / 0.5)
  → sharper than 50/50 but less extreme than 0.7. Use scipy or manual softmax.
- Alternatively, simpler temperature: P_adj = P^(1/T) / (P^(1/T) + (1-P)^(1/T))
- The greedy bracket (temperature=0) must remain identical to current Bracket 1
  output — regression test against existing output before merging.
- Colorado State (12-seed, P=0.66 to beat Memphis) is the strongest upset
  candidate in 2025 and should appear in the upset-path bracket.
- Bracket diversity metric: compute after all brackets generated, just before
  saving to JSON. Add as a "diversity" key in the JSON output.
