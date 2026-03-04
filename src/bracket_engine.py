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


def collect_pod_nodes(region_tree: BracketNode) -> list[BracketNode]:
    """Return all round-2 nodes in left-to-right tree order."""
    pods = []
    def walk(node):
        if node.round_num == 2:
            pods.append(node)
            return
        if node.left: walk(node.left)
        if node.right: walk(node.right)
    walk(region_tree)
    return pods



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
            p_win = sum(node.left.node_probs[t_a.name] * p_b * cache.win_prob(t_b, t_a) for t_a in node.left.teams)
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


def enumerate_pod_combos(
    region_trees: dict[str, BracketNode],
    global_probs: dict[str, dict[int, float]],
    top_k_per_pod: int = 2,
    min_joint_prob: float = 0.0,
) -> tuple[list[dict[str, str]], list[float]]:
    pod_top_teams = {}
    pod_ids = []
    
    # 1. Collect and sort all pods
    for region, tree in region_trees.items():
        pods = collect_pod_nodes(tree)
        for i, pod_node in enumerate(pods):
            pod_id = f"{region}_{i}"
            pod_ids.append(pod_id)
            
            # get probs of winning this pod (reaching round 3)
            team_probs = {t.name: global_probs[t.name][3] for t in pod_node.teams}
            sorted_teams = sorted(team_probs.items(), key=lambda x: x[1], reverse=True)
            pod_top_teams[pod_id] = sorted_teams[:top_k_per_pod]
            
    pools = [pod_top_teams[pid] for pid in pod_ids]
    
    combos = []
    combo_probs = []
    
    # 2. Iterate product
    for combo_tuple in itertools.product(*pools):
        joint_p = 1.0
        combo_dict = {}
        for i, (t, p) in enumerate(combo_tuple):
            joint_p *= p
            combo_dict[pod_ids[i]] = t
            
        if joint_p >= min_joint_prob:
            combos.append(combo_dict)
            combo_probs.append(joint_p)
            
    return combos, combo_probs


def select_diverse_combos(combos: list[dict[str, str]], combo_probs: list[float], n: int, diversity_weight: float = 0.5) -> list[dict[str, str]]:
    import math
    if not combos: return []
    
    best_idx = max(range(len(combo_probs)), key=lambda i: combo_probs[i])
    selected = [combos[best_idx]]
    
    remaining_idxs = set(range(len(combos)))
    remaining_idxs.remove(best_idx)
    
    def dist(c1, c2):
        return sum(1 for r in c1 if c1[r] != c2[r])
        
    log_probs = [math.log(p) if p > 1e-100 else -1000.0 for p in combo_probs]
    
    while len(selected) < n and remaining_idxs:
        best_score = -1.0
        best_i = None
        
        current_rem = list(remaining_idxs)
        max_lp = max(log_probs[i] for i in current_rem)
        min_lp = min(log_probs[i] for i in current_rem)
        lp_range = max_lp - min_lp if max_lp > min_lp else 1.0
        
        for i in current_rem:
            p_score = (log_probs[i] - min_lp) / lp_range
            avg_d = sum(dist(combos[i], s) for s in selected) / len(selected)
            d_score = avg_d / len(combos[i]) 
            
            score = (1 - diversity_weight) * p_score + diversity_weight * d_score
            if score > best_score:
                best_score = score
                best_i = i
                
        selected.append(combos[best_i])
        remaining_idxs.remove(best_i)
        
    return selected


def evaluate_pod_combo_and_build(
    combo: dict[str, str],           # pod_id -> team_name
    pod_nodes: dict[str, BracketNode],  # pod_id -> round-2 node
    region_trees: dict[str, BracketNode],  # region -> root node
    root: BracketNode,
    global_probs: dict[str, dict[int, float]],
    region_matchups: dict,
    final_four_matchups: list[tuple[str, str]],
    all_teams_by_name: dict[str, 'Team'],
) -> 'Bracket':
    import simulate
    bracket = simulate.Bracket()
    
    # 1. Force round 1 & 2 picks by tracing downward from each round-2 pod node
    picks_by_round = {r: [] for r in range(1, 7)}
    for region in region_trees.keys():
        for i in range(4):
            pod_id = f"{region}_{i}"
            forced_winner = combo[pod_id]
            pod_node = pod_nodes[pod_id]
            
            # create a temporary structure to hold the traced picks for this pod
            pod_picks = {r: [None]*(2**(4-r)) for r in range(1, 4)} 
            trace_picks(pod_node, forced_winner, pod_picks)
            
            for r in range(1, 3):
                picks_by_round[r].extend([p for p in pod_picks[r] if p is not None])

    # 2. Greedy EV pass upward for rounds 3, 4, 5, 6
    # Round 3: pod 0 winner vs pod 1 winner (left half), pod 2 inner vs pod 3 winner (right half)
    r3_winners = {} # region -> [left_s16_winner, right_s16_winner]
    r4_nodes = [] # [{'teams': [left_champ, right_champ], ...}]

    for region in region_trees.keys():
        t0 = combo[f"{region}_{0}"]
        t1 = combo[f"{region}_{1}"]
        t2 = combo[f"{region}_{2}"]
        t3 = combo[f"{region}_{3}"]

        ev_0 = global_probs[t0][3] * ROUND_POINTS[3]
        ev_1 = global_probs[t1][3] * ROUND_POINTS[3]
        best_left = t0 if ev_0 >= ev_1 else t1
        
        ev_2 = global_probs[t2][3] * ROUND_POINTS[3]
        ev_3 = global_probs[t3][3] * ROUND_POINTS[3]
        best_right = t2 if ev_2 >= ev_3 else t3
        
        picks_by_round[3].extend([best_left, best_right])
        r3_winners[region] = [best_left, best_right]
        
        # Round 4 (regional final)
        ev_left = global_probs[best_left][4] * ROUND_POINTS[4] 
        ev_right = global_probs[best_right][4] * ROUND_POINTS[4]
        best_r4 = best_left if ev_left >= ev_right else best_right
        
        picks_by_round[4].append(best_r4)
        
        # We need r4_nodes to be parallel to how the original evaluate_combo_and_build passed them to the F4
        r4_nodes.append({'region': region, 'best': best_r4, 'teams': [best_left, best_right], 'scores': {best_left: ev_left, best_right: ev_right}, 'best_score': max(ev_left, ev_right)})


    r5_nodes = []
    # match region 0 with region 1, and region 2 with region 3
    # root.left.left.region_name is final_four_matchups[0][0]
    for i, (left_region, right_region) in enumerate(final_four_matchups):
        left_node = next(n for n in r4_nodes if n['region'] == left_region)
        right_node = next(n for n in r4_nodes if n['region'] == right_region)

        t_left = left_node['best']
        t_right = right_node['best']
        
        # approximate the exact DP score from before by using global_probs[t][5] + node scores
        score_left = left_node['scores'][t_left] if t_left in left_node['scores'] else left_node['best_score']
        score_right = right_node['scores'][t_right] if t_right in right_node['scores'] else right_node['best_score']
        
        s_tleft = global_probs[t_left][5] * 16 + score_left + score_right
        s_tright = global_probs[t_right][5] * 16 + score_right + score_left
        
        best_r5_score = max(s_tleft, s_tright)
        best_r5_team = t_left if s_tleft >= s_tright else t_right
        picks_by_round[5].append(best_r5_team)

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

    picks_by_round[6].append(best_r6_team)
            
    expected_score = sum(global_probs[t][r] * ROUND_POINTS[r] for r, teams in picks_by_round.items() for t in teams)
    bracket.picks = picks_by_round
    bracket.expected_score = expected_score
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
    top_k_per_pod: int = 2,
    diversity_weight: float = 0.7,
) -> tuple[list['Bracket'], dict]:
    root, trees = build_full_tree(region_matchups, final_four_matchups)
    
    global_probs = compute_all_path_probs_tree(root, cache)
    compute_dp(root, global_probs)
    
    # NEW: collect pods and enumerate at pod level
    all_pod_nodes = {}
    for region, tree in trees.items():
        for i, pod_node in enumerate(collect_pod_nodes(tree)):
            pod_id = f"{region}_{i}"
            all_pod_nodes[pod_id] = pod_node

    combos, combo_probs = enumerate_pod_combos(trees, global_probs, top_k_per_pod)
    selected_combos = select_diverse_combos(combos, combo_probs, n_brackets, diversity_weight)
    
    upset_premiums = compute_upset_premiums(region_matchups, global_probs, cache)
    
    out_brackets = []
    for i, combo in enumerate(selected_combos):
        b = evaluate_pod_combo_and_build(
            combo, all_pod_nodes, trees, root, global_probs,
            region_matchups, final_four_matchups, all_teams_by_name
        )
        b.strategy_name = f"Analytical B{i+1}"
        # Summarize forced picks by region for notes
        for region in trees:
            pod_winners = [combo[f"{region}_{j}"] for j in range(4) if f"{region}_{j}" in combo]
            b.notes.append(f"Forced S16 ({region}): {', '.join(pod_winners)}")
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
