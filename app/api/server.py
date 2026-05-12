
import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware

from data.parsers import parse_github_webhook, parse_gitlab_webhook, parse_jenkins_webhook
from data.models import PipelineRun, PipelineStatus, CISource
from app.features.extractor import FeatureExtractor
from app.nlp.classifier import KeywordLogClassifier, LogPreprocessor
from app.rca.engine import RCAEngine
from app.prediction_service import PredictionService   
from app.gitlab_notifier import post_prediction_to_gitlab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipelineiq")


app = FastAPI(
    title="PipelineIQ - AI Powered CI/CD",
    description="Prédiction d'échecs + RCA intelligente",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


prediction_service = PredictionService(model_dir="models")
feature_extractor = FeatureExtractor()
log_preprocessor = LogPreprocessor()
log_classifier = KeywordLogClassifier()
rca_engine = RCAEngine(use_bert=False)   
pipeline_db: Dict[str, Dict] = {}
history_db: List[Dict] = []



def predict_failure(run: PipelineRun, history: list) -> Dict:
    """Fait la prédiction ML complète"""
    features = feature_extractor.extract(run, history)
    
    result = prediction_service.predict(features)
    
    run.predicted_failure_prob = result["failure_probability"]
    run.prediction_features = features
    run.prediction_explanation = result["explanation"]
    
    return result


async def _run_rca_analysis(run: PipelineRun):
    """Analyse RCA en arrière-plan après échec"""
    try:
        rca_report = rca_engine.analyze(
            pipeline_run_id=run.id,
            raw_log=run.raw_logs or "",
            shap_explanation=run.prediction_explanation,
        )
        
        pipeline_db[run.id]["rca"] = {
            "category": rca_report.root_cause_category,
            "confidence": rca_report.confidence,
            "recommendation": rca_report.recommendation,
            "summary": rca_report.llm_summary,
            "evidence": rca_report.evidence[:3],
        }
        logger.info(f"[RCA] {run.id}: {rca_report.root_cause_category} ({rca_report.confidence:.2f})")
    except Exception as e:
        logger.error(f"[RCA] Failed for {run.id}: {e}")


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: Optional[str] = Header(None),
):
    payload = await request.json()
    run = parse_github_webhook(payload)
    if not run:
        return {"status": "ignored"}
    return await _process_pipeline_event(run, background_tasks)


@app.post("/webhook/gitlab")
async def gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    payload = await request.json()
    run = parse_gitlab_webhook(payload)
    if not run:
        return {"status": "ignored"}
    return await _process_pipeline_event(run, background_tasks)


@app.post("/webhook/jenkins")
async def jenkins_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    payload = await request.json()
    run = parse_jenkins_webhook(payload)
    if not run:
        return {"status": "ignored"}
    return await _process_pipeline_event(run, background_tasks)


async def _process_pipeline_event(run: PipelineRun, background_tasks: BackgroundTasks):
    logger.info(f"Processing {run.id} | {run.status} | {run.repo_name}")

    repo_history = [PipelineRun(**r) for r in history_db if r.get("repo_name") == run.repo_name][-200:]

    result = {"run_id": run.id, "status": run.status.value}

    if run.status == PipelineStatus.PENDING:
        prediction = predict_failure(run, repo_history)
        result["prediction"] = {
            "failure_probability": prediction["failure_probability"],
            "risk_level": prediction["risk_level"],
            "top_factors": prediction["explanation"].get("top_factors", []),
            "message": f"Risk: {prediction['risk_level']} ({prediction['failure_probability']:.1%})"
        }
        
        logger.info(f"[PREDICT] {run.id}: {prediction['risk_level']} ({prediction['failure_probability']:.3f})")

        import asyncio
    asyncio.create_task(post_prediction_to_gitlab(run, prediction))

    if run.status == PipelineStatus.FAILED and run.raw_logs:
        background_tasks.add_task(_run_rca_analysis, run)
        result["rca"] = {"status": "queued"}

    run_dict = {
        "id": run.id,
        "source": run.source.value,
        "repo_name": run.repo_name,
        "branch": run.branch,
        "commit_sha": run.commit_sha,
        "status": run.status.value,
        "predicted_failure_prob": run.predicted_failure_prob,
        "triggered_at": run.triggered_at.isoformat(),
    }
    pipeline_db[run.id] = run_dict
    history_db.append(run_dict)

    return result



@app.get("/pipelines/{run_id}")
async def get_pipeline(run_id: str):
    if run_id not in pipeline_db:
        raise HTTPException(status_code=404, detail="Not found")
    return pipeline_db[run_id]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": prediction_service.model is not None,
        "pipelines_tracked": len(pipeline_db),
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("attachments.server:app", host="0.0.0.0", port=8000, reload=True)