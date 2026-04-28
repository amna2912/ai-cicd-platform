# app/main.py - Version corrigée
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Response
import redis
import psycopg2
import psycopg2.extras
import os
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
import uuid

app = FastAPI(title="AI Platform")

class PipelineIngestRequest(BaseModel):
    name: str
    source: str
    status: str
    duration: float
    started_at: str
    finished_at: Optional[str] = None
    logs_url: Optional[str] = None
    metadata: Optional[dict] = {}

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pipeline_name VARCHAR(255) NOT NULL,
                source VARCHAR(50) NOT NULL,
                status VARCHAR(50) NOT NULL,
                duration FLOAT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                finished_at TIMESTAMP,
                logs_url TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_logs_normalized (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
                job_name VARCHAR(255),
                log_level VARCHAR(20),
                category VARCHAR(50),
                message TEXT,
                timestamp TIMESTAMP,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON pipeline_logs_normalized(log_level)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_category ON pipeline_logs_normalized(category)")
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ DB init error: {e}")

init_db()

async def save_pipeline(data: dict) -> str:
    pipeline_id = str(uuid.uuid4())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO pipeline_runs 
        (id, pipeline_name, source, status, duration, started_at, finished_at, logs_url, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        pipeline_id, data["name"], data["source"], data["status"],
        data["duration"], data["started_at"], data.get("finished_at"),
        data.get("logs_url"), psycopg2.extras.Json(data.get("metadata", {}))
    ))
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Pipeline saved: {data['name']} - {data['status']}")
    return pipeline_id

# ==================== WEBHOOK GITHUB ====================
@app.post("/webhook/github")
async def github_webhook(request: Request):
    payload = await request.json()
    print(f"📥 GitHub webhook received")
    
    if request.headers.get("X-GitHub-Event") == "workflow_run":
        workflow = payload.get("workflow_run", {})
        
        # Gestion du status (qui peut être null au début)
        status = workflow.get("conclusion")
        if not status:
            status = workflow.get("status", "unknown")
        if not status:
            status = "unknown"
        
        await save_pipeline({
            "name": workflow.get("name", "unknown"),
            "source": "github-actions",
            "status": status,
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

# ==================== WEBHOOK GITLAB ====================
@app.options("/webhook/gitlab")
async def gitlab_options():
    return Response(
        status_code=200,
        headers={
            "Allow": "GET, HEAD, POST, OPTIONS",
            "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-GitLab-Event",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/webhook/gitlab")
@app.head("/webhook/gitlab")
async def gitlab_webhook_test():
    return {"status": "ok", "message": "GitLab webhook ready"}

@app.post("/webhook/gitlab")
async def gitlab_webhook_post(request: Request):
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
# ==================== WEBHOOK JENKINS ====================
@app.post("/webhook/jenkins")
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

# ==================== API ENDPOINTS ====================
@app.get("/")
def root():
    return {
        "message": "AI Platform Ready 🚀",
        "webhooks": {
            "github": "POST /webhook/github",
            "gitlab": "POST /webhook/gitlab",
            "jenkins": "POST /webhook/jenkins"
        },
        "endpoints": {
            "GET /health": "Health check",
            "GET /api/v1/pipelines": "List pipelines"
        }
    }

@app.get("/health")
def health():
    try:
        conn = get_db_connection()
        conn.close()
        db = "ok"
    except Exception as e:
        db = f"error: {e}"
    
    try:
        r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
        r.ping()
        redis_status = "ok"
    except:
        redis_status = "error"
    
    return {"database": db, "redis": redis_status, "app": "running"}

@app.get("/api/v1/pipelines")
def get_pipelines(limit: int = 100, source: Optional[str] = None):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    query = "SELECT * FROM pipeline_runs"
    params = []
    if source:
        query += " WHERE source = %s"
        params.append(source)
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    
    cur.execute(query, params)
    pipelines = cur.fetchall()
    cur.close()
    conn.close()
    return {"pipelines": pipelines}

@app.get("/api/v1/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: str):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM pipeline_runs WHERE id = %s", (pipeline_id,))
    pipeline = cur.fetchone()
    cur.close()
    conn.close()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)