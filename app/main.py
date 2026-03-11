from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
import redis
import psycopg2
import psycopg2.extras
import os
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
import uuid
import traceback
import requests

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
    """Crée les tables si elles n'existent pas"""
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
                logs_content TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
                job_name VARCHAR(255),
                log_content TEXT,
                log_size INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print(" Database initialized with all tables")
        
    except Exception as e:
        print(f" Erreur initialisation DB: {e}")

init_db()

async def save_pipeline(pipeline_data: dict) -> str:
    """Sauvegarde un pipeline dans PostgreSQL"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        pipeline_id = str(uuid.uuid4())
        
        cur.execute("""
            INSERT INTO pipeline_runs 
            (id, pipeline_name, source, status, duration, started_at, finished_at, logs_url, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            pipeline_id,
            pipeline_data["name"],
            pipeline_data["source"],
            pipeline_data["status"],
            pipeline_data["duration"],
            pipeline_data["started_at"],
            pipeline_data.get("finished_at"),
            pipeline_data.get("logs_url"),
            psycopg2.extras.Json(pipeline_data.get("metadata", {}))
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f" Pipeline sauvegardé: {pipeline_data['name']} - {pipeline_data['status']} (ID: {pipeline_id})")
        return pipeline_id
        
    except Exception as e:
        print(f" Erreur sauvegarde: {e}")
        raise

async def fetch_gitlab_logs(project_id: int, pipeline_id: int):
    """
    Récupère les logs de tous les jobs d'un pipeline GitLab via leur API
    """
    GITLAB_TOKEN = os.getenv("GITLAB_TOKEN", "")
    if not GITLAB_TOKEN:
        print(" GITLAB_TOKEN non configuré dans .env")
        return None
    
    GITLAB_API_URL = f"https://gitlab.com/api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs"
    
    headers = {
        "Authorization": f"Bearer {GITLAB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f" Récupération des jobs pour pipeline {pipeline_id}...")
        
        response = requests.get(GITLAB_API_URL, headers=headers)
        if response.status_code != 200:
            print(f" Erreur récupération jobs: {response.status_code} - {response.text}")
            return None
        
        jobs = response.json()
        print(f" {len(jobs)} jobs trouvés")
        
        all_logs = {}
        for job in jobs:
            job_id = job['id']
            job_name = job['name']
            job_status = job['status']
            
            print(f" Téléchargement logs pour job: {job_name} (statut: {job_status})")
            
            logs_url = f"https://gitlab.com/api/v4/projects/{project_id}/jobs/{job_id}/trace"
            
            logs_response = requests.get(logs_url, headers=headers)
            if logs_response.status_code == 200:
                logs_content = logs_response.text
                all_logs[job_name] = {
                    "content": logs_content,
                    "size": len(logs_content),
                    "status": job_status
                }
                print(f" Logs récupérés: {len(logs_content)} caractères")
            else:
                print(f" Erreur logs job {job_name}: {logs_response.status_code}")
        
        return all_logs
        
    except Exception as e:
        print(f" Erreur fetch GitLab logs: {e}")
        return None

@app.post("/webhook/gitlab")
async def gitlab_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Reçoit les webhooks de GitLab CI ET télécharge automatiquement les logs
    """
    event = request.headers.get("X-GitLab-Event")
    
    try:
        payload = await request.json()
        print(f" Webhook GitLab reçu - Event: {event}")
        print(f" Payload: {payload}")  
    except Exception as e:
        print(f" Erreur lecture payload: {e}")
        return {"status": "error", "message": str(e)}, 400 
    
    if event == "Pipeline Hook":
        pipeline_uuid = await process_gitlab_payload(payload)
        
        project_id = payload.get('project', {}).get('id')
        pipeline_gitlab_id = payload.get('object_attributes', {}).get('id')
        
        if pipeline_uuid and project_id and pipeline_gitlab_id:
            background_tasks.add_task(
                download_and_save_gitlab_logs, 
                pipeline_uuid,
                project_id, 
                pipeline_gitlab_id
            )
            print(f" Tâche de téléchargement des logs planifiée")
        
        return {"status": "received", "pipeline_id": pipeline_uuid}
    
    return {"status": "ignored"}

@app.get("/webhook/gitlab")
@app.head("/webhook/gitlab")
async def gitlab_webhook_test():
    """
    Répond aux requêtes de test (GET/HEAD) de GitLab pour éviter l'erreur 405.
    """
    print(" Requête de test GitLab reçue (GET/HEAD) - Réponse 200 OK")
    return {"status": "ok", "message": "Webhook endpoint is ready for POST requests."}

async def process_gitlab_payload(payload: dict):
    """Traite les données GitLab et retourne l'UUID du pipeline"""
    try:
        print(f" Traitement payload GitLab: {payload}")
        
        if 'object_attributes' in payload:
            object_attrs = payload.get("object_attributes", {})
            project = payload.get("project", {})
            
            pipeline_data = {
                "name": f"{project.get('name', 'unknown')} - Pipeline #{object_attrs.get('id', 'unknown')}",
                "source": "gitlab-ci",
                "status": object_attrs.get("status", "unknown"),
                "duration": object_attrs.get("duration", 0),
                "started_at": object_attrs.get("created_at", datetime.now().isoformat()),
                "finished_at": object_attrs.get("finished_at"),
                "logs_url": object_attrs.get("url"),
                "metadata": {
                    "project": project.get("name"),
                    "project_id": project.get("id"),
                    "branch": object_attrs.get("ref"),
                    "pipeline_id": object_attrs.get("id"),
                    "gitlab_url": object_attrs.get("url"),
                    "source": "gitlab-webhook"
                }
            }
        else:
            pipeline_data = {
                "name": f"gitlab-pipeline-{datetime.now().timestamp()}",
                "source": "gitlab-ci",
                "status": "unknown",
                "duration": 0,
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "logs_url": None,
                "metadata": {"raw_payload": str(payload)[:500]}
            }
        
        print(f" Données préparées: {pipeline_data['name']} - {pipeline_data['status']}")
        return await save_pipeline(pipeline_data)
        
    except Exception as e:
        print(f" Erreur GitLab: {e}")
        traceback.print_exc()
        return None

async def download_and_save_gitlab_logs(pipeline_uuid: str, project_id: int, pipeline_gitlab_id: int):
    """
    Télécharge les logs GitLab et les sauvegarde dans la base
    """
    try:
        print(f" Téléchargement des logs pour pipeline {pipeline_uuid} (GitLab ID: {pipeline_gitlab_id})")
        
        logs_data = await fetch_gitlab_logs(project_id, pipeline_gitlab_id)
        
        if logs_data:
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE pipeline_runs 
                SET metadata = metadata || %s
                WHERE id = %s
            """, (
                psycopg2.extras.Json({
                    "logs_jobs": list(logs_data.keys()),
                    "logs_downloaded": datetime.now().isoformat()
                }),
                pipeline_uuid
            ))
            
            for job_name, job_data in logs_data.items():
                cur.execute("""
                    INSERT INTO pipeline_logs 
                    (pipeline_run_id, job_name, log_content, log_size)
                    VALUES (%s, %s, %s, %s)
                """, (
                    pipeline_uuid,
                    job_name,
                    job_data['content'][:10000],  
                    job_data['size']
                ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            total_size = sum(job['size'] for job in logs_data.values())
            print(f" Logs sauvegardés: {len(logs_data)} jobs, {total_size} caractères")
        else:
            print(f" Aucun log récupéré pour pipeline {pipeline_uuid}")
            
    except Exception as e:
        print(f" Erreur téléchargement logs: {e}")
        traceback.print_exc()
        
@app.get("/api/v1/pipelines/{pipeline_id}/logs")
def get_pipeline_logs(pipeline_id: str):
    """
    Récupère les logs d'un pipeline spécifique
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, pipeline_name, source, status, metadata->>'logs_jobs' as jobs
            FROM pipeline_runs 
            WHERE id = %s
        """, (pipeline_id,))
        
        pipeline = cur.fetchone()
        if not pipeline:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Pipeline not found")
        cur.execute("""
            SELECT job_name, log_size, created_at 
            FROM pipeline_logs 
            WHERE pipeline_run_id = %s
            ORDER BY created_at
        """, (pipeline_id,))
        
        logs_summary = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "pipeline": pipeline,
            "logs": logs_summary,
            "total_logs": len(logs_summary)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines/{pipeline_id}/logs/{job_name}")
def get_job_logs(pipeline_id: str, job_name: str):
    """
    Récupère le contenu des logs d'un job spécifique
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT log_content 
            FROM pipeline_logs 
            WHERE pipeline_run_id = %s AND job_name = %s
        """, (pipeline_id, job_name))
        
        log = cur.fetchone()
        cur.close()
        conn.close()
        
        if not log:
            raise HTTPException(status_code=404, detail="Logs not found")
        
        return {"content": log['log_content']}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    event = request.headers.get("X-GitHub-Event")
    
    try:
        payload = await request.json()
        print(f" Webhook GitHub reçu - Event: {event}")
    except Exception as e:
        return {"status": "error"}, 400
    
    if event == "workflow_run":
        background_tasks.add_task(process_github_payload, payload)
        return {"status": "received"}
    elif event == "ping":
        return {"status": "pong"}
    return {"status": "ignored"}

async def process_github_payload(payload: dict):
    try:
        workflow = payload.get("workflow_run", {})
        
        pipeline_data = {
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
        }
        await save_pipeline(pipeline_data)
    except Exception as e:
        print(f" Erreur GitHub: {e}")

@app.post("/webhook/jenkins")
async def jenkins_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        print(f" Webhook Jenkins reçu")
    except Exception as e:
        return {"status": "error"}, 400
    
    background_tasks.add_task(process_jenkins_payload, payload)
    return {"status": "received"}

async def process_jenkins_payload(payload: dict):
    try:
        name = payload.get("name", "unknown")
        build = payload.get("build", {})
        
        pipeline_data = {
            "name": name,
            "source": "jenkins",
            "status": build.get("status", "unknown").lower(),
            "duration": build.get("duration", 0) / 1000,
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
            "logs_url": f"{build.get('url', '')}console",
            "metadata": {
                "job_name": name,
                "build_number": build.get("number"),
                "build_url": build.get("url")
            }
        }
        await save_pipeline(pipeline_data)
    except Exception as e:
        print(f" Erreur Jenkins: {e}")

@app.get("/")
def root():
    return {
        "message": "AI Platform Running ",
        "webhooks": {
            "github": "POST /webhook/github",
            "gitlab": "POST /webhook/gitlab", 
            "jenkins": "POST /webhook/jenkins"
        },
        "endpoints": {
            "GET /health": "Health check",
            "GET /api/v1/pipelines": "Liste des pipelines",
            "GET /api/v1/pipelines/{id}/logs": "Voir les logs d'un pipeline",
            "POST /api/v1/pipelines/ingest": "Ajout manuel"
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
    except Exception as e:
        redis_status = f"error: {e}"
    
    gitlab_token = os.getenv("GITLAB_TOKEN", "")
    gitlab_status = "ok" if gitlab_token else "warning (no token)"
    
    return {
        "database": db, 
        "redis": redis_status, 
        "gitlab": gitlab_status,
        "app": "running"
    }

@app.post("/api/v1/pipelines/ingest")
async def ingest_pipeline(pipeline: PipelineIngestRequest):
    try:
        pipeline_id = await save_pipeline(pipeline.dict())
        return {"message": "Pipeline saved", "id": pipeline_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines")
def get_pipelines(limit: int = 100, source: Optional[str] = None):
    try:
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("SELECT * FROM pipeline_runs WHERE id = %s", (pipeline_id,))
        pipeline = cur.fetchone()
        cur.close()
        conn.close()
        
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return pipeline
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)