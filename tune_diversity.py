import sys
sys.path.insert(0, './src')
import data_prep
import predict
import bracket_engine
import simulate

def main():
    stats = data_prep.load_team_stats()
    models, scaler = predict.load_models()
    # Ensure SeedNum is not in features
    predict.config.FEATURES = [f for f in predict.config.FEATURES if f != "SeedNum"]
    cache = simulate.MatchupCache(models, scaler, predict.config.FEATURES)
    
    bracket_data = simulate.load_bracket_file('data/bracket_2025.json')
    region_matchups, final_four_matchups, all_teams_by_name = simulate.build_region_matchups_from_file(bracket_data, stats, 2025)
    root, trees = bracket_engine.build_full_tree(region_matchups, final_four_matchups)
    global_probs = bracket_engine.compute_all_path_probs_tree(root, cache)
    
    weights_to_test = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    print("Testing diversity weights...")
    for w in weights_to_test:
        brackets, _ = bracket_engine.run_analytical(
            2025, region_matchups, final_four_matchups, all_teams_by_name, cache, 
            n_brackets=5, diversity_weight=w
        )
        diversity = simulate.compute_bracket_diversity(brackets)
        mean_overlap = diversity['mean_overlap']
        min_overlap = diversity['min_overlap']
        
        # Calculate expected score for Bracket 1 for reference
        score = sum(bracket_engine.ROUND_POINTS[r] for r in range(1, 7) for t in brackets[0].picks[r])
        
        print(f"Weight: {w:.2f} | Mean Overlap: {mean_overlap*100:.1f}% | Min Overlap: {min_overlap*100:.1f}% | B1 Score: {score}")

if __name__ == '__main__':
    main()
