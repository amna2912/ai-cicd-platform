from fastapi import APIRouter, Request, BackgroundTasks
from datetime import datetime
import os
from app.models.database import save_pipeline  

router = APIRouter()

@router.api_route("/webhook/gitlab", methods=["GET", "HEAD", "POST", "OPTIONS"])
async def gitlab_webhook(request: Request, background_tasks: BackgroundTasks):
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return {"status": "ok"}
    
    payload = await request.json()
    print(f"📥 GitLab webhook received")
    
    if request.headers.get("X-GitLab-Event") == "Pipeline Hook":
        object_attrs = payload.get("object_attributes", {})
        project = payload.get("project", {})
        
        pipeline_data = {
            "name": f"{project.get('name', 'unknown')} - Pipeline #{object_attrs.get('id')}",
            "source": "gitlab-ci",
            "status": object_attrs.get("status", "unknown"),
            "duration": object_attrs.get("duration", 0),
            "started_at": object_attrs.get("created_at", datetime.now().isoformat()),
            "finished_at": object_attrs.get("finished_at"),
            "logs_url": object_attrs.get("url"),
            "metadata": {"project_id": project.get("id"), "pipeline_id": object_attrs.get("id")}
        }
        
        await save_pipeline(pipeline_data)
        return {"status": "received"}
    
    return {"status": "ignored"}

@router.post("/webhook/github")
async def github_webhook(request: Request):
    payload = await request.json()
    print(f"📥 GitHub webhook received")
    
    if request.headers.get("X-GitHub-Event") == "workflow_run":
        workflow = payload.get("workflow_run", {})
        await save_pipeline({
            "name": workflow.get("name", "unknown"),
            "source": "github-actions",
            "status": workflow.get("conclusion", "unknown"),
            "duration": 0,
            "started_at": workflow.get("created_at", datetime.now().isoformat()),
            "finished_at": workflow.get("updated_at"),
            "logs_url": f"{workflow.get('html_url', '')}/logs",
            "metadata": {
                "repository": payload.get("repository", {}).get("full_name"),
                "branch": workflow.get("head_branch"),
                "commit": workflow.get("head_sha"),
                "run_id": workflow.get("id")
            }
        })
        return {"status": "received"}
    return {"status": "ignored"}

@router.post("/webhook/jenkins")
async def jenkins_webhook(request: Request):
    payload = await request.json()
    print(f"📥 Jenkins webhook received")
    
    build = payload.get("build", {})
    await save_pipeline({
        "name": payload.get("name", "unknown"),
        "source": "jenkins",
        "status": build.get("status", "unknown").lower(),
        "duration": build.get("duration", 0) / 1000,
        "started_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "logs_url": f"{build.get('url', '')}console",
        "metadata": {"job_name": payload.get("name"), "build_number": build.get("number")}
    })
    return {"status": "received"}