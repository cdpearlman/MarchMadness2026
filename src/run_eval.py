"""Quick script to run evaluation and save results."""
import sys, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')

from data_prep import build_training_data
from feature_engineering import prepare_features
from models import loso_cv, print_results, train_final_models, compute_shap_importance

df = build_training_data()
X, y, seasons = prepare_features(df)

# LOSO CV
results = loso_cv(X, y, seasons, model_name='all')
print_results(results)

# Save per-model means
for name, res_df in results.items():
    res_df.to_csv(f'data/processed/{name}_cv_results.csv', index=False)

# Train final models & SHAP
models, scaler = train_final_models(X, y)
importance = compute_shap_importance(models['xgboost'], X, scaler)
importance.to_csv('data/processed/shap_importance.csv', index=False)

print("\nTop-10 SHAP features:")
print(importance.head(10).to_string(index=False))
