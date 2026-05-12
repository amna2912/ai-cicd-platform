
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from .models import PipelineRun, PipelineStatus, CISource


def _make_id(source: str, external_id: str) -> str:
    """Create a unique internal ID from source + external ID."""
    return f"{source}_{external_id}"


# ─────────────────────────────────────────────
# GITHUB ACTIONS PARSER
# ─────────────────────────────────────────────
def parse_github_webhook(payload: Dict[str, Any]) -> Optional[PipelineRun]:
    """
    Parse a GitHub 'workflow_run' webhook event.

    GitHub sends this when a workflow is: queued, in_progress, completed.
    We capture it at 'queued' stage for PREDICTION (before it runs).

    Payload structure (simplified):
    {
      "action": "queued",           ← this is the key: pipeline just queued!
      "workflow_run": {
        "id": 9876543,
        "status": "queued",
        "head_sha": "abc123...",
        "head_branch": "main",
        "head_commit": {
          "message": "fix: login bug",
          "author": { "email": "dev@co.com", "name": "Alice" }
        },
        "repository": { "full_name": "myorg/myrepo" },
        "created_at": "2025-01-15T10:30:00Z",
        "run_started_at": null,
        "updated_at": "2025-01-15T10:30:01Z",
        "conclusion": null    ← null because not finished yet
      },
      "repository": { ... }
    }
    """
    action = payload.get("action")
    run    = payload.get("workflow_run", {})

    if not run:
        return None

    # Map GitHub status → our internal status
    status_map = {
        "queued":      PipelineStatus.PENDING,
        "in_progress": PipelineStatus.RUNNING,
        "completed":   PipelineStatus.SUCCESS if run.get("conclusion") == "success"
                       else PipelineStatus.FAILED
    }
    status = status_map.get(run.get("status", ""), PipelineStatus.PENDING)

    # Parse timestamps
    triggered_at = datetime.fromisoformat(
        run["created_at"].replace("Z", "+00:00")
    ) if run.get("created_at") else datetime.utcnow()

    started_at = datetime.fromisoformat(
        run["run_started_at"].replace("Z", "+00:00")
    ) if run.get("run_started_at") else None

    finished_at = datetime.fromisoformat(
        run["updated_at"].replace("Z", "+00:00")
    ) if run.get("updated_at") and run.get("conclusion") else None

    # Duration (only if finished)
    duration = None
    if started_at and finished_at:
        duration = int((finished_at - started_at).total_seconds())

    commit = run.get("head_commit", {})
    author = commit.get("author", {})

    return PipelineRun(
        id              = _make_id("github", str(run["id"])),
        source          = CISource.GITHUB,
        external_id     = str(run["id"]),
        repo_name       = run.get("repository", {}).get("full_name", ""),
        branch          = run.get("head_branch", ""),
        commit_sha      = run.get("head_sha", ""),
        commit_message  = commit.get("message", ""),
        author_email    = author.get("email", ""),
        author_name     = author.get("name", ""),
        triggered_at    = triggered_at,
        started_at      = started_at,
        finished_at     = finished_at,
        status          = status,
        duration_seconds= duration,
        raw_payload     = payload,
    )


# ─────────────────────────────────────────────
# GITLAB CI PARSER
# ─────────────────────────────────────────────
def parse_gitlab_webhook(payload: Dict[str, Any]) -> Optional[PipelineRun]:
    """
    Parse a GitLab 'Pipeline Hook' webhook event.

    GitLab sends this when pipeline status changes.
    We capture 'pending' for PREDICTION.

    Payload structure:
    {
      "object_kind": "pipeline",
      "object_attributes": {
        "id": 31,
        "status": "pending",     ← pipeline just created, not running yet
        "created_at": "2025-01-15 10:30:00 UTC",
        "finished_at": null,
        "duration": null,
        "ref": "main"
      },
      "commit": {
        "id": "bcef1a...",
        "message": "feat: new endpoint",
        "author": { "name": "Bob", "email": "bob@co.com" }
      },
      "project": { "path_with_namespace": "myorg/myrepo" }
    }
    """
    if payload.get("object_kind") != "pipeline":
        return None

    attrs  = payload.get("object_attributes", {})
    commit = payload.get("commit", {})
    author = commit.get("author", {})
    project= payload.get("project", {})

    status_map = {
        "pending": PipelineStatus.PENDING,
        "running": PipelineStatus.RUNNING,
        "success": PipelineStatus.SUCCESS,
        "failed":  PipelineStatus.FAILED,
        "canceled":PipelineStatus.CANCELED,
    }
    status = status_map.get(attrs.get("status", ""), PipelineStatus.PENDING)

    def parse_gl_dt(s):
        if not s: return None
        for fmt in ["%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S.%f%z"]:
            try:
                return datetime.strptime(s.replace(" UTC", " +0000"), fmt)
            except ValueError:
                pass
        return datetime.utcnow()

    triggered_at = parse_gl_dt(attrs.get("created_at")) or datetime.utcnow()
    finished_at  = parse_gl_dt(attrs.get("finished_at"))

    return PipelineRun(
        id              = _make_id("gitlab", str(attrs["id"])),
        source          = CISource.GITLAB,
        external_id     = str(attrs["id"]),
        repo_name       = project.get("path_with_namespace", ""),
        branch          = attrs.get("ref", ""),
        commit_sha      = commit.get("id", ""),
        commit_message  = commit.get("message", ""),
        author_email    = author.get("email", ""),
        author_name     = author.get("name", ""),
        triggered_at    = triggered_at,
        started_at      = None,
        finished_at     = finished_at,
        status          = status,
        duration_seconds= attrs.get("duration"),
        raw_payload     = payload,
    )


# ─────────────────────────────────────────────
# JENKINS PARSER
# ─────────────────────────────────────────────
def parse_jenkins_webhook(payload: Dict[str, Any]) -> Optional[PipelineRun]:
    """
    Parse a Jenkins notification plugin payload.
    Install: "Jenkins Notification Plugin" → configure endpoint.

    Payload structure:
    {
      "name": "my-pipeline",
      "build": {
        "number": 42,
        "phase": "STARTED",    ← STARTED = before running, FINALIZED = done
        "status": "SUCCESS",
        "scm": {
          "branch": "main",
          "commit": "def456...",
          "url": "https://github.com/myorg/myrepo"
        },
        "timestamp": 1705312200000,
        "duration": 0
      }
    }
    """
    build  = payload.get("build", {})
    scm    = build.get("scm", {})
    phase  = build.get("phase", "").upper()

    phase_to_status = {
        "STARTED":   PipelineStatus.PENDING,
        "COMPLETED": PipelineStatus.RUNNING,
        "FINALIZED": PipelineStatus.SUCCESS if build.get("status") == "SUCCESS"
                     else PipelineStatus.FAILED,
    }
    status = phase_to_status.get(phase, PipelineStatus.PENDING)

    # Jenkins timestamp is milliseconds
    ts_ms = build.get("timestamp", 0)
    triggered_at = datetime.utcfromtimestamp(ts_ms / 1000) if ts_ms else datetime.utcnow()

    duration_ms = build.get("duration", 0)
    duration = int(duration_ms / 1000) if duration_ms else None

    # Extract repo name from scm URL
    scm_url   = scm.get("url", "")
    repo_name = scm_url.rstrip("/").split("/")[-2] + "/" + scm_url.rstrip("/").split("/")[-1] \
                if "/" in scm_url else scm_url

    pipeline_name  = payload.get("name", "unknown")
    build_number   = str(build.get("number", "0"))
    external_id    = f"{pipeline_name}_{build_number}"

    return PipelineRun(
        id              = _make_id("jenkins", external_id),
        source          = CISource.JENKINS,
        external_id     = external_id,
        repo_name       = repo_name,
        branch          = scm.get("branch", ""),
        commit_sha      = scm.get("commit", ""),
        commit_message  = "",   # Jenkins doesn't always send this
        author_email    = "",
        author_name     = "",
        triggered_at    = triggered_at,
        started_at      = triggered_at if phase == "STARTED" else None,
        finished_at     = None,
        status          = status,
        duration_seconds= duration,
        raw_payload     = payload,
    )