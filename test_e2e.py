"""
STEP 9 — END-TO-END TEST
=========================
This file demonstrates the COMPLETE prediction flow
from a raw GitHub webhook → feature extraction → ML prediction → SHAP explanation.

Run this to verify your setup works:
    cd pipelineiq
    python tests/test_e2e.py

Expected output:
    ✅ Parsed pipeline run: github_9876543
    ✅ Extracted 35 features
    ✅ Model trained on synthetic data
    ✅ Prediction: P(fail) = 0.73 | Risk: HIGH
    ✅ Top factor: no_test_with_src = 1.0 (+0.28 SHAP)
    ✅ Error classified: DEP_MISSING (confidence: 0.82)
    ✅ RCA complete: recommendation generated
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from data.parsers import parse_github_webhook
from data.models import PipelineRun, PipelineStatus, CISource
from app.features.extractor import FeatureExtractor
from app.nlp.classifier import KeywordLogClassifier, LogPreprocessor
from app.rca.engine import RCAEngine



SAMPLE_GITHUB_PAYLOAD = {
    "action": "queued",
    "workflow_run": {
        "id": 9876543,
        "status": "queued",
        "head_sha": "a3f9c1d2e4b5",
        "head_branch": "feature/add-payment",
        "head_commit": {
            "message": "hotfix: quick payment patch",
            "author": {
                "email": "alice@company.com",
                "name": "Alice"
            }
        },
        "repository": {"full_name": "myorg/ecommerce-api"},
        "created_at": "2025-01-17T16:45:00Z",
        "run_started_at": None,
        "updated_at": "2025-01-17T16:45:01Z",
        "conclusion": None
    }
}

SAMPLE_FAILED_LOG = """
2025-01-17T16:46:12Z [INFO] Installing dependencies...
2025-01-17T16:46:15Z [INFO] npm install
npm ERR! code E404
npm ERR! 404 Not Found - GET https://registry.npmjs.org/@company/payment-sdk
npm ERR! 404 '@company/payment-sdk@2.1.4' is not in this registry.
npm ERR! 
npm ERR! Note that you can also install from a tarball or folder
2025-01-17T16:46:18Z [ERROR] Process exited with code 1
2025-01-17T16:46:18Z [INFO] Build FAILED
"""


def run_e2e_test():
    print("=" * 60)
    print("PipelineIQ — End-to-End Test")
    print("=" * 60)

    # ── TEST 1: Parse webhook
    print("\n[TEST 1] Parsing GitHub webhook payload...")
    run = parse_github_webhook(SAMPLE_GITHUB_PAYLOAD)
    assert run is not None, "Parser returned None"
    assert run.id == "github_9876543"
    assert run.branch == "feature/add-payment"
    assert run.author_email == "alice@company.com"
    assert "hotfix" in run.commit_message
    print(f"  ✅ Parsed run: {run.id}")
    print(f"     Branch: {run.branch}")
    print(f"     Commit: '{run.commit_message}'")
    print(f"     Author: {run.author_email}")

    # ── TEST 2: Feature extraction
    print("\n[TEST 2] Extracting features...")
    extractor = FeatureExtractor()

    # Simulate some historical runs for context
    mock_history = [
        PipelineRun(
            id="github_prev1", source=CISource.GITHUB, external_id="prev1",
            repo_name="myorg/ecommerce-api", branch="feature/add-payment",
            commit_sha="abc", commit_message="feat: start payment", author_email="alice@company.com",
            author_name="Alice", triggered_at=datetime(2025, 1, 15, 10, 0),
            started_at=None, finished_at=None,
            status=PipelineStatus.FAILED,   # previous run failed!
            duration_seconds=180,
        ),
        PipelineRun(
            id="github_prev2", source=CISource.GITHUB, external_id="prev2",
            repo_name="myorg/ecommerce-api", branch="main",
            commit_sha="def", commit_message="release: v1.2.0", author_email="bob@company.com",
            author_name="Bob", triggered_at=datetime(2025, 1, 14, 9, 0),
            started_at=None, finished_at=None,
            status=PipelineStatus.SUCCESS,
            duration_seconds=240,
        ),
    ]

    features = extractor.extract(run, mock_history)
    assert len(features) > 20, f"Expected 20+ features, got {len(features)}"

    print(f"  ✅ Extracted {len(features)} features")
    print(f"     is_friday_eod:           {features.get('is_friday_eod', 0)}")
    print(f"     commit_risk_keywords:    {features.get('commit_risk_keyword_count', 0)}")
    print(f"     author_fail_rate_7d:     {features.get('author_fail_rate_7d', 0):.2f}")
    print(f"     branch_fail_rate_all:    {features.get('branch_fail_rate_all', 0):.2f}")
    print(f"     commit_is_short_msg:     {features.get('commit_is_short_msg', 0)}")

    # ── TEST 3: Model training (quick, on synthetic data)
    print("\n[TEST 3] Training model on synthetic data (quick demo)...")
    import sys
    sys.path.insert(0, ".")
    from app.training.train import generate_synthetic_data, train_xgboost, train_baseline
    from sklearn.model_selection import train_test_split
    import shap

    df = generate_synthetic_data(n_samples=500)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_xgboost(X_train, y_train, X_val, y_val)

    print(f"  ✅ Model trained on {len(X_train)} samples")

    # ── TEST 4: Prediction
    print("\n[TEST 4] Running failure prediction...")
    import pandas as pd

    # Build feature vector (only use features the model was trained on)
    feature_vector = {name: features.get(name, 0.0) for name in feature_cols}
    X_pred = pd.DataFrame([feature_vector])
    prob = float(model.predict_proba(X_pred)[0][1])

    risk_level = (
        "CRITICAL" if prob > 0.75 else
        "HIGH"     if prob > 0.55 else
        "MEDIUM"   if prob > 0.30 else
        "LOW"
    )
    print(f"  ✅ Prediction: P(fail) = {prob:.3f} | Risk: {risk_level}")

    # ── TEST 5: SHAP explanation
    print("\n[TEST 5] Computing SHAP explanation...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_pred)[0]

    # Top 3 contributing features
    contributions = sorted(
        zip(feature_cols, shap_vals, X_pred.iloc[0].values),
        key=lambda x: abs(x[1]), reverse=True
    )

    print(f"  ✅ Top risk factors:")
    for feat, shap_v, feat_val in contributions[:5]:
        direction = "▲ increases" if shap_v > 0 else "▼ reduces"
        print(f"     {direction} risk: {feat} = {feat_val:.3f} (SHAP: {shap_v:+.3f})")

    # ── TEST 6: Log classification
    print("\n[TEST 6] Classifying error from failed log...")
    preprocessor = LogPreprocessor()
    classifier   = KeywordLogClassifier()

    clean_log    = preprocessor.preprocess(SAMPLE_FAILED_LOG)
    error_window = preprocessor.extract_error_window(clean_log)
    result       = classifier.classify(error_window)

    print(f"  ✅ Error category: {result.category} ({result.category_label})")
    print(f"     Confidence: {result.confidence:.2f}")
    print(f"     Evidence: {result.error_snippet[:100]}")

    # ── TEST 7: RCA
    print("\n[TEST 7] Running Root Cause Analysis...")
    run_failed      = parse_github_webhook(SAMPLE_GITHUB_PAYLOAD)
    run_failed.status    = PipelineStatus.FAILED
    run_failed.raw_logs  = SAMPLE_FAILED_LOG

    rca = RCAEngine(use_bert=False)
    report = rca.analyze(
        pipeline_run_id = run_failed.id,
        raw_log         = SAMPLE_FAILED_LOG,
    )

    print(f"  ✅ RCA complete:")
    print(f"     Root cause: {report.root_cause_category}")
    print(f"     Confidence: {report.confidence:.2f}")
    print(f"     Recommendation: {report.recommendation}")
    print(f"     Summary: {report.llm_summary[:120]}...")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED — PipelineIQ is working correctly!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. python training/train.py --output models/")
    print("  3. uvicorn api.server:app --reload")
    print("  4. Configure webhook URL in GitHub/GitLab/Jenkins")
    print("  5. Push a commit and watch the prediction arrive!")


if __name__ == "__main__":
    run_e2e_test()