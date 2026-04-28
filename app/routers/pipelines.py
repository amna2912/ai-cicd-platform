from fastapi import APIRouter, HTTPException
from typing import Optional
from app.models.database import get_db_connection  
import psycopg2.extras

router = APIRouter()

@router.get("/api/v1/pipelines")
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

@router.get("/api/v1/pipelines/{pipeline_id}")
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