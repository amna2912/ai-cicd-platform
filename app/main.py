from fastapi import FastAPI, HTTPException, BackgroundTasks
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
    """This defines the shape of data we expect from CI/CD tools"""
    name: str              
    source: str            
    status: str            
    duration: float        
    started_at: str        
    finished_at: Optional[str] = None 
    logs_url: Optional[str] = None      
    metadata: Optional[dict] = {}       

def get_db_connection():
    """Simple function to connect to PostgreSQL"""
    return psycopg2.connect(os.getenv("DATABASE_URL"))
def init_db():
    """Create table if it doesn't exist"""
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
    
    conn.commit()
    cur.close()
    conn.close()
    print(" Database initialized")

init_db()

def analyze_pipeline_background(pipeline_id: str):
    """This runs AFTER we send response to user (doesn't block)"""
    print(f" Analyzing pipeline {pipeline_id} in background...")
    print(f" Analysis done for {pipeline_id}")

@app.get("/")
def root():
    """Welcome endpoint - shows API is working"""
    return {
        "message": "AI Platform is RUNNING!",
        "time": datetime.now().isoformat(),
        "endpoints": {
            "POST /api/v1/pipelines/ingest": "Add new pipeline data",
            "GET /api/v1/pipelines": "List all pipelines",
            "GET /health": "Check if services are healthy"
        }
    }

@app.get("/health")
def health():
    """Check if database and redis are working"""
    try:
        conn = get_db_connection()
        conn.close()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    try:
        r = redis.from_url(os.getenv("REDIS_URL"))
        r.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    return {
        "database": db_status,
        "redis": redis_status,
        "app": "running"
    }

@app.post("/api/v1/pipelines/ingest")
async def ingest_pipeline(pipeline: PipelineIngestRequest, background_tasks: BackgroundTasks):
    """
    MAIN ENDPOINT: Receive data from CI/CD tools
    - Gets pipeline data
    - Saves to database
    - Triggers background analysis
    - Returns success message
    """
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
            pipeline.name,
            pipeline.source,
            pipeline.status,
            pipeline.duration,
            pipeline.started_at,
            pipeline.finished_at,
            pipeline.logs_url,
            psycopg2.extras.Json(pipeline.metadata)  
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        background_tasks.add_task(analyze_pipeline_background, pipeline_id)
        
        return {
            "message": " Pipeline saved successfully",
            "pipeline_id": pipeline_id,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/v1/pipelines")
def get_pipelines(limit: int = 100):
    """Get all pipelines (simplified)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT id, pipeline_name, source, status, duration, created_at
            FROM pipeline_runs
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        
        pipelines = cur.fetchall()
        cur.close()
        conn.close()
        
        return {"pipelines": pipelines}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: str):
    """Get one specific pipeline by ID"""
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)