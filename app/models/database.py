import psycopg2
import psycopg2.extras
import os
import uuid

def get_db_connection():
    """Retourne une connexion à la base de données"""
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
        print(f"❌ Erreur initialisation DB: {e}")

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
        
        print(f"✅ Pipeline sauvegardé: {pipeline_data['name']} - {pipeline_data['status']}")
        return pipeline_id
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        raise