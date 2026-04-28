from fastapi import APIRouter
from app.models.database import get_db_connection  
import psycopg2.extras

router = APIRouter()

@router.get("/api/v1/analytics/performance")
def analyze_performance(): 
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    cur.execute("""
        SELECT 
            source,
            AVG(duration) as avg_duration,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'failed') as failures
        FROM pipeline_runs
        GROUP BY source
    """)
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return {"avg_duration_by_source": results, "slow_pipelines": [], "bottlenecks": []}