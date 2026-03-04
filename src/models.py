"""
Model training, evaluation, and feature importance.

Models:
  - Logistic Regression (calibrated probabilities)
  - XGBoost (raw predictive power)
  - Random Forest (regularisation, diversity)
  - Ensemble (weighted average of predicted probabilities)

Evaluation:
  - Leave-One-Season-Out (LOSO) cross-validation
  - Metrics: log-loss (primary), accuracy, AUC-ROC
  - SHAP feature importance for XGBoost
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ───────────────────────────────────────────────────────────────
# Model factories
# ───────────────────────────────────────────────────────────────

def make_logistic() -> LogisticRegression:
    return LogisticRegression(**config.LOGISTIC_PARAMS)


def make_xgboost() -> xgb.XGBClassifier:
    params = {k: v for k, v in config.XGBOOST_PARAMS.items()
              if k != "early_stopping_rounds"}
    return xgb.XGBClassifier(**params)


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)


# ───────────────────────────────────────────────────────────────
# Leave-One-Season-Out cross-validation
# ───────────────────────────────────────────────────────────────

def loso_cv(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    model_name: str = "all",
) -> dict[str, pd.DataFrame]:
    """
    Run Leave-One-Season-Out CV for specified model(s).

    Parameters
    ----------
    X : features (diff_* columns)
    y : labels (0/1)
    seasons : season for each row
    model_name : 'logistic', 'xgboost', 'rf', 'ensemble', or 'all'

    Returns
    -------
    dict mapping model_name -> DataFrame of per-season metrics
    """
    unique_seasons = sorted(seasons.unique())
    models_to_run = _get_models(model_name)

    results: dict[str, list[dict]] = {name: [] for name in models_to_run}

    for test_season in unique_seasons:
        train_mask = seasons != test_season
        test_mask = seasons == test_season

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Scale features (important for LogReg; doesn't hurt tree models)
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

        # Train each model
        trained: dict[str, object] = {}
        probas: dict[str, np.ndarray] = {}

        for name in models_to_run:
            if name == "ensemble":
                continue  # computed after individual models

            model = _create_model(name)

            if name == "xgboost":
                model.fit(
                    X_train_s, y_train,
                    eval_set=[(X_test_s, y_test)],
                    verbose=False,
                )
            else:
                model.fit(X_train_s, y_train)

            trained[name] = model
            probas[name] = model.predict_proba(X_test_s)[:, 1]

        # Ensemble: weighted average
        if "ensemble" in models_to_run:
            component_names = [n for n in ["logistic", "xgboost", "rf"] if n in probas]
            weights = config.ENSEMBLE_WEIGHTS[: len(component_names)]
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]

            ens_proba = np.zeros(len(y_test))
            for name, w in zip(component_names, weights):
                ens_proba += w * probas[name]

            probas["ensemble"] = ens_proba

        # Compute metrics
        for name in models_to_run:
            if name not in probas:
                continue
            p = probas[name]
            preds = (p >= 0.5).astype(int)
            results[name].append({
                "season": test_season,
                "log_loss": log_loss(y_test, p),
                "accuracy": accuracy_score(y_test, preds),
                "auc_roc": roc_auc_score(y_test, p),
                "n_games": len(y_test) // 2,
            })

    return {name: pd.DataFrame(rows) for name, rows in results.items()}


def loso_collect_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Run LOSO CV and collect all out-of-fold predictions for each base model.

    Returns (y_all, model_probas) where y_all is the concatenated true labels
    and model_probas maps model_name -> concatenated predicted probabilities,
    both in the same row order.
    """
    unique_seasons = sorted(seasons.unique())
    base_models = ["logistic", "xgboost", "rf"]

    y_parts: list[np.ndarray] = []
    proba_parts: dict[str, list[np.ndarray]] = {n: [] for n in base_models}

    for test_season in unique_seasons:
        train_mask = seasons != test_season
        test_mask = seasons == test_season

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index,
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )

        for name in base_models:
            model = _create_model(name)
            if name == "xgboost":
                model.fit(X_train_s, y_train,
                          eval_set=[(X_test_s, y_test)], verbose=False)
            else:
                model.fit(X_train_s, y_train)
            proba_parts[name].append(model.predict_proba(X_test_s)[:, 1])

        y_parts.append(y_test.values)

    y_all = np.concatenate(y_parts)
    model_probas = {n: np.concatenate(proba_parts[n]) for n in base_models}
    return y_all, model_probas


def optimize_ensemble_weights(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
) -> list[float]:
    """
    Find ensemble weights that minimize log-loss on out-of-fold predictions.
    Uses Nelder-Mead on the simplex (softmax parameterization).
    """
    names = ["logistic", "xgboost", "rf"]
    proba_matrix = np.column_stack([model_probas[n] for n in names])

    def objective(raw_weights: np.ndarray) -> float:
        exp_w = np.exp(raw_weights - raw_weights.max())
        w = exp_w / exp_w.sum()
        blended = proba_matrix @ w
        blended = np.clip(blended, 1e-15, 1 - 1e-15)
        return log_loss(y_true, blended)

    best_result = None
    for init in [np.zeros(3), np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.0, 0.5])]:
        result = minimize(objective, init, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    exp_w = np.exp(best_result.x - best_result.x.max())
    optimal = exp_w / exp_w.sum()

    print(f"\n  Ensemble weight optimization (minimize LOSO log-loss):")
    for n, w in zip(names, optimal):
        print(f"    {n}: {w:.4f}")
    print(f"    Optimized log-loss: {best_result.fun:.6f}")

    old_w = np.array(config.ENSEMBLE_WEIGHTS[:3])
    old_w = old_w / old_w.sum()
    old_blend = proba_matrix @ old_w
    old_ll = log_loss(y_true, np.clip(old_blend, 1e-15, 1 - 1e-15))
    print(f"    Previous  log-loss: {old_ll:.6f}")
    print(f"    Improvement: {old_ll - best_result.fun:.6f}")

    return [round(float(w), 4) for w in optimal]


def fit_calibrators(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
) -> dict[str, IsotonicRegression]:
    """
    Fit isotonic regression calibrators on LOSO out-of-fold predictions
    for each base model plus the ensemble blend.
    """
    calibrators: dict[str, IsotonicRegression] = {}
    names = ["logistic", "xgboost", "rf"]

    for name in names:
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(model_probas[name], y_true)
        calibrators[name] = iso

    weights = np.array(config.ENSEMBLE_WEIGHTS[:3])
    weights = weights / weights.sum()
    proba_matrix = np.column_stack([model_probas[n] for n in names])
    ensemble_raw = proba_matrix @ weights

    iso_ens = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso_ens.fit(ensemble_raw, y_true)
    calibrators["ensemble"] = iso_ens

    raw_ll = log_loss(y_true, np.clip(ensemble_raw, 1e-15, 1 - 1e-15))
    cal_ll = log_loss(y_true, iso_ens.predict(ensemble_raw))
    print(f"\n  Probability calibration (isotonic regression on LOSO OOF):")
    print(f"    Ensemble raw  log-loss: {raw_ll:.6f}")
    print(f"    Ensemble cal  log-loss: {cal_ll:.6f}")
    print(f"    Improvement: {raw_ll - cal_ll:.6f}")

    return calibrators


def _get_models(model_name: str) -> list[str]:
    all_models = ["logistic", "xgboost", "rf", "ensemble"]
    if model_name == "all":
        return all_models
    elif model_name == "ensemble":
        return all_models  # need components for ensemble
    else:
        return [model_name]


def _create_model(name: str):
    if name == "logistic":
        return make_logistic()
    elif name == "xgboost":
        return make_xgboost()
    elif name == "rf":
        return make_random_forest()
    else:
        raise ValueError(f"Unknown model: {name}")


# ───────────────────────────────────────────────────────────────
# Train final models on all data
# ───────────────────────────────────────────────────────────────

def train_final_models(
    X: pd.DataFrame, y: pd.Series
) -> tuple[dict[str, object], StandardScaler]:
    """
    Train all models on the full dataset. Returns trained models + scaler.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )

    models = {}
    for name in ["logistic", "xgboost", "rf"]:
        model = _create_model(name)
        model.fit(X_scaled, y)
        models[name] = model
        print(f"  Trained {name}")

    return models, scaler


def ensemble_predict(
    models: dict[str, object],
    scaler: StandardScaler,
    X: pd.DataFrame,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Predict probabilities using the weighted ensemble."""
    if weights is None:
        weights = config.ENSEMBLE_WEIGHTS

    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=X.columns, index=X.index
    )

    component_names = ["logistic", "xgboost", "rf"]
    w_sum = sum(weights[: len(component_names)])
    weights = [w / w_sum for w in weights[: len(component_names)]]

    proba = np.zeros(len(X))
    for name, w in zip(component_names, weights):
        proba += w * models[name].predict_proba(X_scaled)[:, 1]

    return proba


# ───────────────────────────────────────────────────────────────
# SHAP feature importance
# ───────────────────────────────────────────────────────────────

def compute_shap_importance(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    scaler: StandardScaler,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Compute SHAP values for the XGBoost model."""
    import shap

    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=X.columns, index=X.index
    )

    if len(X_scaled) > max_samples:
        X_scaled = X_scaled.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    importance = pd.DataFrame({
        "feature": X_scaled.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance


# ───────────────────────────────────────────────────────────────
# Pretty-print results
# ───────────────────────────────────────────────────────────────

def print_results(results: dict[str, pd.DataFrame]) -> None:
    """Print summary of LOSO CV results."""
    print(f"\n{'='*70}")
    print(f"LOSO Cross-Validation Results")
    print(f"{'='*70}")

    for name, df in results.items():
        avg_ll = df["log_loss"].mean()
        avg_acc = df["accuracy"].mean()
        avg_auc = df["auc_roc"].mean()
        print(f"\n  {name.upper()}")
        print(f"    Log-loss:  {avg_ll:.4f}")
        print(f"    Accuracy:  {avg_acc:.4f} ({avg_acc*100:.1f}%)")
        print(f"    AUC-ROC:   {avg_auc:.4f}")

    # Per-season detail for best model
    best_name = min(results, key=lambda n: results[n]["log_loss"].mean())
    print(f"\n  Best model by log-loss: {best_name.upper()}")
    print(f"\n  Per-season breakdown ({best_name}):")
    best_df = results[best_name]
    for _, row in best_df.iterrows():
        print(
            f"    {int(row['season'])}: "
            f"LL={row['log_loss']:.4f}  "
            f"Acc={row['accuracy']:.3f}  "
            f"AUC={row['auc_roc']:.3f}  "
            f"({int(row['n_games'])} games)"
        )


# ───────────────────────────────────────────────────────────────
# CLI entry point
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_prep import build_training_data
    from feature_engineering import prepare_features

    print("Step 1: Building training data...")
    training_df = build_training_data()

    print("\nStep 2: Computing features...")
    X, y, seasons = prepare_features(training_df)

    print("\nStep 3: Running LOSO Cross-Validation...")
    results = loso_cv(X, y, seasons, model_name="all")
    print_results(results)

    print("\nStep 4: Optimizing ensemble weights...")
    y_oof, model_probas = loso_collect_oof_predictions(X, y, seasons)
    optimal_weights = optimize_ensemble_weights(y_oof, model_probas)
    print(f"\n  To apply: set ENSEMBLE_WEIGHTS = {optimal_weights} in config.py")

    print("\nStep 5: Fitting probability calibrators...")
    calibrators = fit_calibrators(y_oof, model_probas)

    print("\nStep 6: Training final models...")
    models, scaler = train_final_models(X, y)

    print("\nStep 7: Saving models + calibrators...")
    import pickle
    model_dir = config.PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "trained_models.pkl", "wb") as f:
        pickle.dump({"models": models, "scaler": scaler, "calibrators": calibrators}, f)
    print(f"  Saved to {model_dir / 'trained_models.pkl'}")

    print("\nStep 8: Feature importance (SHAP)...")
    importance = compute_shap_importance(models["xgboost"], X, scaler)
    print(f"\n  Top-10 features by SHAP importance:")
    print(importance.head(10).to_string(index=False))

    shap_path = config.PROCESSED_DIR / "shap_importance.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(shap_path, index=False)
    print(f"  Saved SHAP importance -> {shap_path}")
