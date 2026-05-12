

import os
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from typing import Optional


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression   # baseline
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def load_training_data(csv_path: str) -> pd.DataFrame:
    """
    Load pipeline run history from CSV.
    CSV has one row per pipeline run, columns = features + 'label'.

    Expected columns (all numeric):
      files_changed_count, lines_added, lines_deleted, ...
      label  ← 1 = failed, 0 = success  (this is what we predict)

    In practice you generate this CSV from your database:
        SELECT features..., CASE WHEN status='failed' THEN 1 ELSE 0 END as label
        FROM pipeline_runs
        WHERE status IN ('success', 'failed')
    """
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} samples")
    print(f"[DATA] Class distribution:\n{df['label'].value_counts()}")
    print(f"[DATA] Features: {[c for c in df.columns if c != 'label']}")
    return df


def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate realistic synthetic training data for DEVELOPMENT & TESTING.
    In production, replace this with real historical data from your database.

    This function simulates what your real feature matrix looks like.
    It encodes real-world patterns:
      - Friday EOD + no tests + large change → high failure probability
      - Small change + test files changed + stable branch → low failure prob
    """
    np.random.seed(42)
    n = n_samples

    data = {
        "files_changed_count":    np.random.poisson(6, n).astype(float),
        "lines_added":            np.random.exponential(100, n),
        "lines_deleted":          np.random.exponential(50, n),
        "has_test_changes":       np.random.binomial(1, 0.55, n).astype(float),
        "test_file_ratio":        np.random.beta(2, 4, n),
        "has_ci_config_change":   np.random.binomial(1, 0.12, n).astype(float),
        "has_infra_change":       np.random.binomial(1, 0.08, n).astype(float),
        "no_test_with_src":       np.random.binomial(1, 0.30, n).astype(float),
        "large_change_flag":      np.random.binomial(1, 0.20, n).astype(float),
        "log_lines_changed":      np.random.normal(4.5, 1.5, n),

        "author_fail_rate_all":   np.random.beta(2, 6, n),
        "author_fail_rate_7d":    np.random.beta(2, 5, n),
        "author_consecutive_fails": np.random.poisson(0.5, n).astype(float),
        "author_total_runs":      np.random.poisson(50, n).astype(float),

        "branch_fail_rate_7d":    np.random.beta(2, 5, n),
        "branch_is_main":         np.random.binomial(1, 0.30, n).astype(float),
        "branch_is_feature":      np.random.binomial(1, 0.55, n).astype(float),
        "days_since_branch_fail": np.random.exponential(10, n),

        "cache_enabled":          np.random.binomial(1, 0.60, n).astype(float),
        "num_jobs":                np.random.randint(1, 8, n).astype(float),
        "has_test_stage":         np.random.binomial(1, 0.80, n).astype(float),

        "trigger_hour":           np.random.randint(0, 24, n).astype(float),
        "is_friday":              np.random.binomial(1, 0.20, n).astype(float),
        "is_friday_eod":          np.random.binomial(1, 0.05, n).astype(float),
        "is_weekend":             np.random.binomial(1, 0.10, n).astype(float),

        "commit_risk_keyword_count": np.random.poisson(0.8, n).astype(float),
        "commit_is_short_msg":    np.random.binomial(1, 0.25, n).astype(float),
        "commit_has_conventional":np.random.binomial(1, 0.45, n).astype(float),
    }

    df = pd.DataFrame(data)

    
    failure_score = (
        0.5  * df["author_fail_rate_7d"]         # strong signal
      + 0.4  * df["branch_fail_rate_7d"]          # strong signal
      + 0.3  * df["no_test_with_src"]             # risky pattern
      + 0.25 * df["has_ci_config_change"]         # risky: config touched
      + 0.2  * df["large_change_flag"]            # big changes = risky
      + 0.15 * df["is_friday_eod"]                # Friday afternoon
      + 0.15 * df["commit_risk_keyword_count"] / 5
      + 0.1  * df["author_consecutive_fails"] / 3
      - 0.2  * df["has_test_changes"]             # tests = safer
      - 0.15 * df["cache_enabled"]                # cache = more stable
      - 0.1  * df["commit_has_conventional"]      # disciplined = safer
      + np.random.normal(0, 0.15, n)             # noise
    )

    prob = 1 / (1 + np.exp(-2 * failure_score))   # sigmoid
    df["label"] = (prob > np.random.uniform(0, 1, n)).astype(int)

    print(f"[SYNTHETIC] Generated {n} samples")
    print(f"[SYNTHETIC] Failure rate: {df['label'].mean():.1%}")
    return df


def train_baseline(X_train, y_train):
    """
    Logistic Regression baseline.
    We need this to PROVE our XGBoost model is actually better.
    Without a baseline, you can't claim your ML adds value.
    This is required for academic rigor.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost classifier — our main model.

    KEY HYPERPARAMETERS EXPLAINED:
      n_estimators:   number of trees (more = slower but better, until overfit)
      max_depth:      how deep each tree grows (3-6 is good for tabular data)
      learning_rate:  step size (smaller = more trees needed but better generalization)
      scale_pos_weight: handles class imbalance (if 70% pass, 30% fail → weight = 2.33)
      subsample:      fraction of data per tree (prevents overfitting)
      colsample_bytree: fraction of features per tree (prevents overfitting)
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"[XGB] Class imbalance ratio: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators       = 300,
        max_depth          = 5,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = scale_pos_weight,
        min_child_weight   = 5,
        gamma              = 0.1,
        reg_alpha          = 0.1,       # L1 regularization
        reg_lambda         = 1.0,       # L2 regularization
        use_label_encoder  = False,
        eval_metric        = "auc",
        early_stopping_rounds = 30,
        random_state       = 42,
        n_jobs             = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set     = [(X_val, y_val)],
        verbose      = 50,
    )

    best_iter = model.best_iteration
    print(f"[XGB] Best iteration: {best_iter}")
    return model



def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Full evaluation of the model.
    Reports precision, recall, F1, AUC-ROC.

    For a failure prediction model:
      - HIGH RECALL is more important than precision
        (better to warn falsely than to miss a real failure)
      - AUC-ROC > 0.75 is a good baseline
      - AUC-ROC > 0.85 is excellent for CI/CD prediction
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION: {model_name}")
    print('='*50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"             Predicted Pass  Predicted Fail")
    print(f"Actual Pass:      {cm[0][0]:5d}          {cm[0][1]:5d}")
    print(f"Actual Fail:      {cm[1][0]:5d}          {cm[1][1]:5d}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\nFalse Negative Rate (missed failures): {fn/(fn+tp):.1%}")
    print(f"False Positive Rate (false alarms):    {fp/(fp+tn):.1%}")

    return {"auc": auc, "report": classification_report(y_test, y_pred, output_dict=True)}



def compute_shap_explainer(model, X_train: pd.DataFrame, output_dir: str):
    """
    SHAP (SHapley Additive exPlanations) — explains WHY the model made each prediction.

    This is what makes our prediction "explainable AI" rather than a black box.

    For each prediction we can say:
      "Risk score = 0.82 because:
        +0.35 → author fail rate is 45% (above average)
        +0.28 → no test files changed
        +0.15 → CI config file was modified
        -0.10 → cache is enabled (reduces risk)"

    This is CRUCIAL for developer trust. Nobody acts on a black-box score.
    """
    print("\n[SHAP] Computing SHAP values (this may take 30-60 seconds)...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_train,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_importance.png"), dpi=150)
    plt.close()
    print(f"[SHAP] Saved feature importance plot")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, max_display=20, show=False)
    plt.title("SHAP Feature Impact (direction + magnitude)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), dpi=150)
    plt.close()
    print(f"[SHAP] Saved beeswarm plot")

    return explainer


def explain_single_prediction(
    explainer, X_sample: pd.DataFrame, feature_names: list
) -> dict:
    """
    For a SINGLE pipeline run, explain which features drove the prediction.
    This is what gets returned in the API response so the developer can act.

    Returns a dict like:
    {
      "risk_score": 0.82,
      "top_factors": [
        {"feature": "author_fail_rate_7d",  "value": 0.45, "shap": +0.35, "direction": "increases_risk"},
        {"feature": "no_test_with_src",     "value": 1.0,  "shap": +0.28, "direction": "increases_risk"},
        {"feature": "cache_enabled",        "value": 1.0,  "shap": -0.10, "direction": "reduces_risk"},
      ]
    }
    """
    shap_vals = explainer.shap_values(X_sample)[0]   # shape: (n_features,)
    feature_vals = X_sample.iloc[0].to_dict()

    contributions = [
        {
            "feature":   feat,
            "value":     round(float(feature_vals[feat]), 4),
            "shap":      round(float(shap_val), 4),
            "direction": "increases_risk" if shap_val > 0 else "reduces_risk"
        }
        for feat, shap_val in zip(feature_names, shap_vals)
    ]

    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "top_factors": contributions[:5],    # return top 5 most impactful features
        "all_factors": contributions,
    }


def save_model(model, explainer, feature_names: list, metrics: dict, output_dir: str):
    """
    Save everything needed to serve the model.
    We save: the model itself, the SHAP explainer, feature list, metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "failure_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    explainer_path = os.path.join(output_dir, "shap_explainer.pkl")
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)

    meta = {
        "feature_names": feature_names,
        "metrics": metrics,
        "model_type": "XGBoostClassifier",
        "version": "1.0",
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVE] Model saved to: {model_path}")
    print(f"[SAVE] Explainer saved to: {explainer_path}")
    return model_path


def train_pipeline(data_path: Optional[str] = None, output_dir: str = "models"):
    """
    Full end-to-end training pipeline.
    Call this once to train, then your API loads the saved model.
    """
    print("=" * 60)
    print("PIPELINEIQ — ML Training Pipeline")
    print("=" * 60)

    # ── Load data
    if data_path and os.path.exists(data_path):
        df = load_training_data(data_path)
    else:
        print("[INFO] No data file found, using synthetic data for demo")
        df = generate_synthetic_data(n_samples=3000)

    # ── Separate features and label
    LABEL_COL = "label"
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_cols]
    y = df[LABEL_COL]

    # ── Train/val/test split (60/20/20)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
    )
    print(f"\n[SPLIT] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Train baseline (Logistic Regression)
    print("\n[1/3] Training baseline (Logistic Regression)...")
    lr_model = train_baseline(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression (Baseline)")

    # ── Train main model (XGBoost)
    print("\n[2/3] Training main model (XGBoost)...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Main Model)")

    # ── Improvement summary
    print(f"\n[IMPROVEMENT] Baseline AUC: {lr_metrics['auc']:.4f}")
    print(f"[IMPROVEMENT] XGBoost AUC:  {xgb_metrics['auc']:.4f}")
    print(f"[IMPROVEMENT] Delta:        +{xgb_metrics['auc'] - lr_metrics['auc']:.4f}")

    # ── SHAP explainability
    print("\n[3/3] Computing SHAP explainability...")
    os.makedirs(output_dir, exist_ok=True)
    explainer = compute_shap_explainer(xgb_model, X_train, output_dir)

    # ── Save
    model_path = save_model(
        xgb_model, explainer, feature_cols, xgb_metrics, output_dir
    )

    print("\n✅ Training complete!")
    print(f"   Model AUC-ROC: {xgb_metrics['auc']:.4f}")
    print(f"   Files saved to: {output_dir}/")

    return xgb_model, explainer, feature_cols


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PipelineIQ failure prediction model")
    parser.add_argument("--data",   type=str, default=None,     help="Path to training CSV")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    train_pipeline(data_path=args.data, output_dir=args.output)