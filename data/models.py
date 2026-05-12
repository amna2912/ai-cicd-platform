
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum


class CISource(str, Enum):
    GITHUB  = "github"
    GITLAB  = "gitlab"
    JENKINS = "jenkins"


class PipelineStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    FAILED   = "failed"
    CANCELED = "canceled"


@dataclass
class PipelineRun:
    """
    Unified pipeline run record — source-agnostic.
    One of these is created for every pipeline from every source.
    Stored in your database (PostgreSQL / SQLite).
    """
    # ── Identity
    id: str                              # e.g. "github_123456"
    source: CISource                     # where it came from
    external_id: str                     # original ID from GitHub/GitLab/Jenkins

    # ── Repository context
    repo_name: str                       # e.g. "myorg/myrepo"
    branch: str                          # e.g. "feature/add-auth"
    commit_sha: str                      # e.g. "a3f9c1..."
    commit_message: str
    author_email: str
    author_name: str

    # ── Timing
    triggered_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    # ── Outcome
    status: PipelineStatus
    duration_seconds: Optional[int]

    # ── Raw data (kept for NLP / log analysis)
    raw_logs: Optional[str] = None       # full log text
    raw_payload: Optional[Dict] = None   # original webhook JSON

    # ── File changes (extracted from commit API)
    files_changed: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0

    # ── ML prediction (filled after model runs)
    predicted_failure_prob: Optional[float] = None
    prediction_features: Optional[Dict]     = None
    prediction_explanation: Optional[Dict]  = None  # SHAP values


@dataclass
class PipelineStep:
    """
    Individual step/job inside a pipeline.
    GitHub calls them 'steps', GitLab 'jobs', Jenkins 'stages'.
    """
    pipeline_run_id: str
    step_name: str
    status: PipelineStatus
    duration_seconds: Optional[int]
    logs: Optional[str] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None