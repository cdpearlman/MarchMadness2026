# Decision Record

<!-- Append-only. Record significant decisions with reasoning. -->

## Analytical DP over Monte Carlo for bracket optimization
**Date**: pre-2026-03-03 (documented at bootstrap)
**Context**: Needed a bracket optimization strategy that could generate diverse, strategically sound brackets
**Options considered**: Monte Carlo simulation vs. analytical dynamic programming
**Decision**: Moved to analytical DP (`bracket_engine.py`)
**Reasoning**: Monte Carlo was too greedy — always favored the higher-probability outcome in each game (e.g., 51% > 49%) without considering the strategic value of upsets (better downstream matchups, bracket diversity, pool scoring upside). The DP approach evaluates full paths through the bracket and can weigh diversity.
**Revisit if**: DP approach becomes too slow for larger bracket counts or if a hybrid approach could combine DP's strategic depth with MC's sampling flexibility

## Pre-tournament stats only (no in-tournament data)
**Date**: pre-2026-03-03 (documented at bootstrap)
**Context**: Choosing which stat variants to use as features
**Options considered**: Full-season stats vs. pre-tournament stats
**Decision**: Pre-tournament stats only (`Pre-Tournament.AdjOE`, etc.)
**Reasoning**: Using stats that include tournament game results would be data leakage — the model would be "seeing" outcomes it's trying to predict
**Revisit if**: Never — this is a fundamental correctness constraint

## Potential switch to Barttorvik data source
**Date**: 2026-03-03 (under consideration)
**Context**: Current KenPom data comes from Kaggle datasets which may lag behind the season
**Options considered**: Keep KenPom/Kaggle vs. switch to Barttorvik (barttorvik.com)
**Decision**: Pending investigation
**Reasoning**: Data freshness is the primary concern — need current-season stats for 2026 predictions
**Revisit if**: Decision needed before 2026 tournament bracket generation
