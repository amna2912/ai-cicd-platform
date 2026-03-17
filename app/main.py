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
from utils.normalizer import LogNormalizer

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
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_logs_bruts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
                job_name VARCHAR(255),
                log_content TEXT,
                log_size INT,
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
        print("✅ Database initialized with all tables")
        
    except Exception as e:
        print(f"❌ Erreur initialisation DB: {e}")

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
        
        print(f"✅ Pipeline sauvegardé: {pipeline_data['name']} - {pipeline_data['status']} (ID: {pipeline_id})")
        return pipeline_id
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        raise

@app.post("/webhook/gitlab")
async def gitlab_webhook(request: Request, background_tasks: BackgroundTasks):
    """Reçoit les webhooks de GitLab CI ET télécharge automatiquement les logs"""
    event = request.headers.get("X-GitLab-Event")
    
    try:
        payload = await request.json()
        print(f"📥 Webhook GitLab reçu - Event: {event}")
    except Exception as e:
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
            print(f"🔄 Tâche de téléchargement des logs planifiée")
        
        return {"status": "received", "pipeline_id": pipeline_uuid}
    
    return {"status": "ignored"}

@app.get("/webhook/gitlab")
@app.head("/webhook/gitlab")
async def gitlab_webhook_test():
    """Répond aux requêtes de test (GET/HEAD) de GitLab"""
    return {"status": "ok", "message": "GitLab webhook ready"}

async def process_gitlab_payload(payload: dict):
    """Traite les données GitLab et retourne l'UUID du pipeline"""
    try:
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
                    "gitlab_url": object_attrs.get("url")
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
        
        return await save_pipeline(pipeline_data)
        
    except Exception as e:
        print(f"❌ Erreur GitLab: {e}")
        traceback.print_exc()
        return None

async def fetch_gitlab_logs(project_id: int, pipeline_id: int):
    """Récupère les logs de tous les jobs d'un pipeline GitLab via leur API"""
    GITLAB_TOKEN = os.getenv("GITLAB_TOKEN", "")
    if not GITLAB_TOKEN:
        print("⚠️ GITLAB_TOKEN non configuré")
        return None
    
    headers = {"Authorization": f"Bearer {GITLAB_TOKEN}"}
    jobs_url = f"https://gitlab.com/api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs"
    
    try:
        jobs_response = requests.get(jobs_url, headers=headers)
        if jobs_response.status_code != 200:
            return None
        
        jobs = jobs_response.json()
        all_logs = {}
        
        for job in jobs:
            job_name = job['name']
            logs_url = f"https://gitlab.com/api/v4/projects/{project_id}/jobs/{job['id']}/trace"
            logs_response = requests.get(logs_url, headers=headers)
            
            if logs_response.status_code == 200:
                all_logs[job_name] = {
                    "content": logs_response.text,
                    "size": len(logs_response.text),
                    "status": job['status']
                }
        
        return all_logs
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

async def download_and_save_gitlab_logs(pipeline_uuid: str, project_id: int, pipeline_gitlab_id: int):
    """Télécharge les logs GitLab et les sauvegarde (normalisés)"""
    try:
        logs_data = await fetch_gitlab_logs(project_id, pipeline_gitlab_id)
        
        if logs_data:
            conn = get_db_connection()
            cur = conn.cursor()
            
            for job_name, job_data in logs_data.items():
                normalized_entries = LogNormalizer.normalize_logs(job_data['content'], job_name)
                
                for entry in normalized_entries:
                    cur.execute("""
                        INSERT INTO pipeline_logs_normalized 
                        (pipeline_run_id, job_name, log_level, category, message, timestamp, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        pipeline_uuid,
                        entry.get('job_name'),
                        entry.get('level'),
                        entry.get('category'),
                        entry.get('message'),
                        entry.get('timestamp'),
                        psycopg2.extras.Json(entry.get('metadata', {}))
                    ))
            
            conn.commit()
            cur.close()
            conn.close()
            print(f"✅ Logs normalisés sauvegardés pour {pipeline_uuid}")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

@app.get("/api/v1/pipelines/{pipeline_id}/logs/normalized")
def get_normalized_logs(pipeline_id: str, level: Optional[str] = None, category: Optional[str] = None):
    """Récupère les logs normalisés d'un pipeline"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("SELECT id FROM pipeline_runs WHERE id = %s", (pipeline_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        query = "SELECT * FROM pipeline_logs_normalized WHERE pipeline_run_id = %s"
        params = [pipeline_id]
        
        if level:
            query += " AND log_level = %s"
            params.append(level)
        
        if category:
            query += " AND category = %s"
            params.append(category)
        
        query += " ORDER BY timestamp ASC NULLS LAST"
        
        cur.execute(query, params)
        logs = cur.fetchall()
        
        cur.close()
        conn.close()
        
        stats = {"total": len(logs), "by_level": {}, "by_category": {}}
        for log in logs:
            if log['log_level']:
                stats["by_level"][log['log_level']] = stats["by_level"].get(log['log_level'], 0) + 1
            if log['category']:
                stats["by_category"][log['category']] = stats["by_category"].get(log['category'], 0) + 1
        
        return {"pipeline_id": pipeline_id, "stats": stats, "logs": logs}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines/{pipeline_id}/logs/summary")
def get_logs_summary(pipeline_id: str):
    """Résumé des logs pour un pipeline"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE log_level = 'ERROR') as errors,
                COUNT(*) FILTER (WHERE log_level = 'WARNING') as warnings,
                COUNT(DISTINCT category) as categories
            FROM pipeline_logs_normalized 
            WHERE pipeline_run_id = %s
        """, (pipeline_id,))
        
        stats = cur.fetchone()
        
        cur.execute("""
            SELECT category, COUNT(*) as count
            FROM pipeline_logs_normalized 
            WHERE pipeline_run_id = %s AND category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
            LIMIT 5
        """, (pipeline_id,))
        
        top_categories = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "pipeline_id": pipeline_id,
            "summary": stats,
            "top_categories": top_categories
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    event = request.headers.get("X-GitHub-Event")
    try:
        payload = await request.json()
        print(f"📥 Webhook GitHub reçu - Event: {event}")
    except:
        return {"status": "error"}, 400
    
    if event == "workflow_run":
        background_tasks.add_task(process_github_payload, payload)
        return {"status": "received"}
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
        print(f"❌ Erreur GitHub: {e}")

@app.post("/webhook/jenkins")
async def jenkins_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        print(f"📥 Webhook Jenkins reçu")
    except:
        return {"status": "error"}, 400
    
    background_tasks.add_task(process_jenkins_payload, payload)
    return {"status": "received"}

async def process_jenkins_payload(payload: dict):
    try:
        build = payload.get("build", {})
        pipeline_data = {
            "name": payload.get("name", "unknown"),
            "source": "jenkins",
            "status": build.get("status", "unknown").lower(),
            "duration": build.get("duration", 0) / 1000,
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
            "logs_url": f"{build.get('url', '')}console",
            "metadata": {
                "job_name": payload.get("name"),
                "build_number": build.get("number"),
                "build_url": build.get("url")
            }
        }
        await save_pipeline(pipeline_data)
    except Exception as e:
        print(f"❌ Erreur Jenkins: {e}")

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
            "GET /api/v1/pipelines": "Liste des pipelines",
            "GET /api/v1/pipelines/{id}": "Détail d'un pipeline",
            "GET /api/v1/pipelines/{id}/logs/normalized": "Logs normalisés",
            "GET /api/v1/pipelines/{id}/logs/summary": "Résumé des logs"
        }
    }

@app.get("/health")
def health():
    db = "ok" if get_db_connection() else "error"
    redis_status = "ok" if redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379")).ping() else "error"
    gitlab_status = "ok" if os.getenv("GITLAB_TOKEN") else "warning"
    
    return {"database": db, "redis": redis_status, "gitlab": gitlab_status, "app": "running"}

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