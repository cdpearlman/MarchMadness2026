from __future__ import annotations
import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict, Any

if TYPE_CHECKING:
    from simulate import Team, Bracket, MatchupCache

ROUND_POINTS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}

class BracketNode:
    def __init__(self, teams: list['Team'], round_num: int, left: Optional['BracketNode']=None, right: Optional['BracketNode']=None, match_idx: int=0, region_name: str=""):
        self.teams = teams
        self.round_num = round_num
        self.left = left
        self.right = right
        self.match_idx = match_idx
        self.region_name = region_name
        
        # dp tables
        self.score: dict[str, float] = {}
        self.best_unforced_score: float = -1.0
        self.best_unforced_team: Optional[str] = None
        
        # path probs storage
        self.node_probs: dict[str, float] = {}


def build_region_tree(matchups: list[tuple['Team', 'Team']], region_name: str) -> BracketNode:
    current_nodes = []
    for i, (t_a, t_b) in enumerate(matchups):
        leaf = BracketNode([t_a, t_b], 1, None, None, i, region_name)
        current_nodes.append(leaf)
    
    round_num = 2
    while len(current_nodes) > 1:
        next_nodes = []
        for i in range(0, len(current_nodes), 2):
            left = current_nodes[i]
            right = current_nodes[i+1]
            n = BracketNode(left.teams + right.teams, round_num, left, right, i//2, region_name)
            next_nodes.append(n)
        current_nodes = next_nodes
        round_num += 1
        
    return current_nodes[0]


def build_full_tree(region_matchups: dict[str, list[tuple['Team', 'Team']]], final_four_matchups: list[tuple[str, str]]) -> tuple[BracketNode, dict[str, BracketNode]]:
    trees = {}
    for region, matchups in region_matchups.items():
        trees[region] = build_region_tree(matchups, region)
        
    f4_nodes = []
    for i, (r_a, r_b) in enumerate(final_four_matchups):
        left = trees[r_a]
        right = trees[r_b]
        teams = left.teams + right.teams
        node = BracketNode(teams, 5, left, right, i)
        f4_nodes.append(node)
        
    root = BracketNode(f4_nodes[0].teams + f4_nodes[1].teams, 6, f4_nodes[0], f4_nodes[1], 0)
    return root, trees


def compute_all_path_probs_tree(root: BracketNode, cache: 'MatchupCache') -> dict[str, dict[int, float]]:
    reach_probs = defaultdict(lambda: {0: 1.0})
    
    def walk(node: BracketNode):
        if node.left is None and node.right is None:
            t_a, t_b = node.teams[0], node.teams[1]
            p_a = cache.win_prob(t_a, t_b)
            p_b = 1.0 - p_a
            reach_probs[t_a.name][1] = p_a
            reach_probs[t_b.name][1] = p_b
            node.node_probs = {t_a.name: p_a, t_b.name: p_b}
            return
            
        walk(node.left)
        walk(node.right)
        
        node.node_probs = {}
        for t_a in node.left.teams:
            p_a = node.left.node_probs[t_a.name]
            p_win = sum(p_a * node.right.node_probs[t_b.name] * cache.win_prob(t_a, t_b) for t_b in node.right.teams)
            reach_probs[t_a.name][node.round_num] = p_win
            node.node_probs[t_a.name] = p_win
            
        for t_b in node.right.teams:
            p_b = node.right.node_probs[t_b.name]
            p_win = sum(p_a * p_b * cache.win_prob(t_b, t_a) for t_a in node.left.teams)
            reach_probs[t_b.name][node.round_num] = p_win
            node.node_probs[t_b.name] = p_win
            
    walk(root)
    return dict(reach_probs)


def compute_dp(node: BracketNode, global_probs: dict[str, dict[int, float]]):
    if node.left is None and node.right is None:
        for t in node.teams:
            val = global_probs[t.name][1] * ROUND_POINTS[1]
            node.score[t.name] = val
            if val > node.best_unforced_score:
                node.best_unforced_score = val
                node.best_unforced_team = t.name
        return

    compute_dp(node.left, global_probs)
    compute_dp(node.right, global_probs)

    for t in node.teams:
        my_pts = global_probs[t.name][node.round_num] * ROUND_POINTS[node.round_num]
        
        if t in node.left.teams:
            left_val = node.left.score[t.name]
            right_val = node.right.best_unforced_score
        else:
            right_val = node.right.score[t.name]
            left_val = node.left.best_unforced_score
            
        val = my_pts + left_val + right_val
        node.score[t.name] = val
        if val > node.best_unforced_score:
            node.best_unforced_score = val
            node.best_unforced_team = t.name


def trace_picks(node: BracketNode, forced_team: str, picks_by_round: dict[int, list[Optional[str]]]):
    picks_by_round[node.round_num][node.match_idx] = forced_team
    if node.left is None and node.right is None:
        return
        
    if forced_team in [t.name for t in node.left.teams]:
        trace_picks(node.left, forced_team, picks_by_round)
        trace_picks(node.right, node.right.best_unforced_team, picks_by_round)
    else:
        trace_picks(node.right, forced_team, picks_by_round)
        trace_picks(node.left, node.left.best_unforced_team, picks_by_round)


def enumerate_final_four_combos(
    region_path_probs: dict[str, dict[str, float]], 
    top_k_per_region: int = 4,
    min_prob_floor: float = 0.0,
) -> tuple[list[dict[str, str]], list[float]]:
    
    region_top_teams = {}
    for region, probs in region_path_probs.items():
        sorted_teams = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        region_top_teams[region] = sorted_teams[:top_k_per_region]
        
    regions = list(region_path_probs.keys())
    pools = [region_top_teams[r] for r in regions]
    
    combos = []
    combo_probs = []
    
    for combo_tuple in itertools.product(*pools):
        joint_p = 1.0
        combo_dict = {}
        for i, (t, p) in enumerate(combo_tuple):
            joint_p *= p
            combo_dict[regions[i]] = t
            
        if joint_p >= min_prob_floor:
            combos.append(combo_dict)
            combo_probs.append(joint_p)
            
    return combos, combo_probs


def select_diverse_combos(combos: list[dict[str, str]], combo_probs: list[float], n: int, diversity_weight: float = 0.5) -> list[dict[str, str]]:
    if not combos: return []
    
    best_idx = max(range(len(combo_probs)), key=lambda i: combo_probs[i])
    selected = [combos[best_idx]]
    
    remaining_idxs = set(range(len(combos)))
    remaining_idxs.remove(best_idx)
    
    def dist(c1, c2):
        return sum(1 for r in c1 if c1[r] != c2[r])
        
    while len(selected) < n and remaining_idxs:
        best_score = -1.0
        best_i = None
        
        max_p = max(combo_probs[i] for i in remaining_idxs) if remaining_idxs else 1.0
        if max_p == 0: max_p = 1.0 
        
        for i in remaining_idxs:
            p_score = combo_probs[i] / max_p
            avg_d = sum(dist(combos[i], s) for s in selected) / len(selected)
            d_score = avg_d / 4.0 
            
            score = (1.0 - diversity_weight) * p_score + diversity_weight * d_score
            if score > best_score:
                best_score = score
                best_i = i
                
        selected.append(combos[best_i])
        remaining_idxs.remove(best_i)
        
    return selected


def evaluate_combo_and_build(combo: dict[str, str], trees: dict[str, BracketNode], root: BracketNode, global_probs: dict[str, dict[int, float]], region_matchups: dict) -> 'Bracket':
    import simulate
    bracket = simulate.Bracket()
    
    r5_nodes = []
    for f4_node in [root.left, root.right]:
        left_region = f4_node.left.region_name
        right_region = f4_node.right.region_name
        
        t_left = combo[left_region]
        t_right = combo[right_region]
        
        score_left = f4_node.left.score[t_left]
        score_right = f4_node.right.score[t_right]
        
        s_tleft = global_probs[t_left][5] * 16 + score_left + score_right
        s_tright = global_probs[t_right][5] * 16 + score_right + score_left
        
        best_r5_score = max(s_tleft, s_tright)
        best_r5_team = t_left if s_tleft >= s_tright else t_right
        r5_nodes.append({'teams': [t_left, t_right], 'scores': {t_left: s_tleft, t_right: s_tright}, 'best': best_r5_team, 'best_score': best_r5_score})
        
    left_r5 = r5_nodes[0]
    right_r5 = r5_nodes[1]
    
    best_r6_score = -1.0
    best_r6_team = None
    
    for t in left_r5['teams']:
        s = global_probs[t][6] * 32 + left_r5['scores'][t] + right_r5['best_score']
        if s > best_r6_score:
            best_r6_score = s
            best_r6_team = t
            
    for t in right_r5['teams']:
        s = global_probs[t][6] * 32 + right_r5['scores'][t] + left_r5['best_score']
        if s > best_r6_score:
            best_r6_score = s
            best_r6_team = t

    picks_by_round = {r: [] for r in range(1, 7)}
    picks_by_round[6].append(best_r6_team)
    
    if best_r6_team in r5_nodes[0]['teams']:
        t_f4_1 = best_r6_team
        t_f4_2 = r5_nodes[1]['best']
    else:
        t_f4_1 = r5_nodes[0]['best']
        t_f4_2 = best_r6_team
        
    picks_by_round[5].append(t_f4_1)
    picks_by_round[5].append(t_f4_2)
    
    for region in region_matchups.keys():
        tree = trees[region]
        forced_winner = combo[region]
        region_picks = {r: [None]*(2**(4-r)) for r in range(1, 5)}
        trace_picks(tree, forced_winner, region_picks)
        for r in range(1, 5):
            picks_by_round[r].extend(region_picks[r])
            
    bracket.picks = picks_by_round
    bracket.expected_score = best_r6_score
    return bracket


def compute_upset_premiums(
    region_matchups: dict[str, list[tuple['Team', 'Team']]],
    reach_probs: dict[str, dict[int, float]],
    cache: 'MatchupCache',
    min_seed: int = 5,
) -> list[dict]:
    premiums = []
    
    def calc_ev(t_name, start_r):
        return sum(reach_probs.get(t_name, {}).get(r, 0.0) * ROUND_POINTS[r] for r in range(start_r, 7))
        
    for region, matchups in region_matchups.items():
        for t_a, t_b in matchups:
            chalk = t_a if t_a.seed < t_b.seed else t_b
            dog = t_b if t_a.seed < t_b.seed else t_a
            
            if dog.seed >= min_seed:
                ev_dog = calc_ev(dog.name, 1)
                ev_chalk = calc_ev(chalk.name, 1)
                premium = ev_dog - ev_chalk
                
                single_r_ev = reach_probs[dog.name][1] * ROUND_POINTS[1]
                road_description = f"Plays #{chalk.seed} {chalk.name} in R1 (winnable)"
                
                premiums.append({
                    'team': dog,
                    'seed': dog.seed,
                    'opponent': chalk,
                    'round': 1,
                    'single_round_ev': single_r_ev,
                    'cumulative_ev': ev_dog,
                    'upset_premium': premium,
                    'road_description': road_description
                })
                
    premiums.sort(key=lambda x: x['upset_premium'], reverse=True)
    return premiums


def run_analytical(
    season: int,
    region_matchups: dict[str, list[tuple['Team', 'Team']]],
    final_four_matchups: list[tuple[str, str]],
    all_teams_by_name: dict[str, 'Team'],
    cache: 'MatchupCache',
    n_brackets: int = 5,
    top_k_per_region: int = 4,
    diversity_weight: float = 0.5,
) -> tuple[list['Bracket'], dict]:
    root, trees = build_full_tree(region_matchups, final_four_matchups)
    
    global_probs = compute_all_path_probs_tree(root, cache)
    compute_dp(root, global_probs)
    
    region_path_probs = {}
    for region in region_matchups.keys():
        region_path_probs[region] = {t.name: global_probs[t.name][4] for pair in region_matchups[region] for t in pair}
        
    combos, combo_probs = enumerate_final_four_combos(region_path_probs, top_k_per_region)
    selected_combos = select_diverse_combos(combos, combo_probs, n_brackets, diversity_weight)
    
    upset_premiums = compute_upset_premiums(region_matchups, global_probs, cache)
    
    out_brackets = []
    for i, combo in enumerate(selected_combos):
        b = evaluate_combo_and_build(combo, trees, root, global_probs, region_matchups)
        b.strategy_name = f"Analytical B{i+1}"
        b.notes = [f"Forced F4: {', '.join(combo.values())}"]
        out_brackets.append(b)
        
    from simulate import compute_bracket_diversity
    diversity = compute_bracket_diversity(out_brackets)
    
    report = {
        'path_probs': global_probs,
        'upset_premiums': upset_premiums,
        'final_four_combos': combos,
        'selected_combos': selected_combos,
        'diversity_matrix': diversity
    }
    
    return out_brackets, report
