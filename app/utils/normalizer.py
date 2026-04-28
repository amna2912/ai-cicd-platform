import re
from datetime import datetime
from typing import Dict, Any, List

class MetadataNormalizer:
    """
    Normalise les métadonnées de toutes les sources vers un format unifié
    """
    
    @staticmethod
    def normalize_github(metadata: dict) -> dict:
        """Normalise les métadonnées GitHub"""
        return {
            "vcs": {
                "repository": metadata.get("repository"),
                "branch": metadata.get("branch"),
                "commit_hash": metadata.get("commit"),
                "commit_short": metadata.get("commit", "")[:7] if metadata.get("commit") else None,
            },
            "ci": {
                "provider": "github-actions",
                "run_id": metadata.get("run_id"),
                "run_number": metadata.get("run_number"),
                "run_attempt": metadata.get("run_attempt"),
                "triggered_by": metadata.get("triggered_by"),
            },
            "build": {},
            "test_results": {}
        }
    
    @staticmethod
    def normalize_gitlab(metadata: dict) -> dict:
        """Normalise les métadonnées GitLab"""
        return {
            "vcs": {
                "repository": metadata.get("project"),
                "branch": metadata.get("branch"),
                "commit_hash": None,  
            },
            "ci": {
                "provider": "gitlab-ci",
                "pipeline_id": metadata.get("pipeline_id"),
                "project_id": metadata.get("project_id"),
                "ci_url": metadata.get("gitlab_url"),
            },
            "build": {},
            "test_results": {}
        }
    
    @staticmethod
    def normalize_jenkins(metadata: dict) -> dict:
        """Normalise les métadonnées Jenkins"""
        return {
            "vcs": {
                "repository": None,
                "branch": None,
            },
            "ci": {
                "provider": "jenkins",
                "job_name": metadata.get("job_name"),
                "build_number": metadata.get("build_number"),
                "build_url": metadata.get("build_url"),
            },
            "build": {},
            "test_results": {}
        }
    
    @staticmethod
    def normalize(source: str, metadata: dict) -> dict:
        """Point d'entrée principal pour la normalisation"""
        if source == "github-actions":
            return MetadataNormalizer.normalize_github(metadata)
        elif source == "gitlab-ci":
            return MetadataNormalizer.normalize_gitlab(metadata)
        elif source == "jenkins":
            return MetadataNormalizer.normalize_jenkins(metadata)
        else:
            return {"raw": metadata}


class LogNormalizer:
    """
    Normalise les logs bruts en entrées structurées
    """
    
    ERROR_PATTERNS = {
        "database": r"(?i)(database|db|postgres|mysql|connection timeout|sql)",
        "network": r"(?i)(network|timeout|connection refused|unreachable)",
        "memory": r"(?i)(memory|out of memory|oom|heap)",
        "permission": r"(?i)(permission|access denied|unauthorized|403|401)",
        "syntax": r"(?i)(syntax error|unexpected token|compile error)",
        "dependency": r"(?i)(dependency|module not found|cannot find package|404)",
    }
    
    @classmethod
    def parse_log_line(cls, line: str, job_name: str = None) -> Dict[str, Any]:
        """
        Transforme une ligne de log en structure normalisée
        """
        line = line.strip()
        if not line:
            return None
        
        level = "INFO"
        if any(word in line.upper() for word in ["ERROR", "❌", "FAIL", "EXCEPTION"]):
            level = "ERROR"
        elif any(word in line.upper() for word in ["WARN", "⚠️", "WARNING"]):
            level = "WARNING"
        elif any(word in line.upper() for word in ["DEBUG"]):
            level = "DEBUG"
        
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', line)
        timestamp = timestamp_match.group(0) if timestamp_match else None
        
        category = "other"
        if level == "ERROR" or level == "WARNING":
            for cat, pattern in cls.ERROR_PATTERNS.items():
                if re.search(pattern, line):
                    category = cat
                    break
        
        return {
            "timestamp": timestamp,
            "level": level,
            "category": category if level in ["ERROR", "WARNING"] else None,
            "job_name": job_name,
            "message": line[:500],  
            "has_error": level == "ERROR",
            "metadata": {
                "length": len(line),
                "word_count": len(line.split()),
            }
        }
    
    @classmethod
    def normalize_logs(cls, logs_content: str, job_name: str = None) -> List[Dict]:
        """Normalise tout un contenu de logs"""
        normalized = []
        for line in logs_content.split('\n'):
            parsed = cls.parse_log_line(line, job_name)
            if parsed:
                normalized.append(parsed)
        return normalized