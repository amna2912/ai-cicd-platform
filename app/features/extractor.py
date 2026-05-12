
import re
import math
from datetime import datetime
from typing import Dict, List, Optional, Any
from data.models import PipelineRun, PipelineStatus


# ─────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────

class FeatureExtractor:
    """
    Takes a PipelineRun + historical context from DB,
    returns a feature dict ready for the ML model.
    """

    def extract(
        self,
        run: PipelineRun,
        history: List[PipelineRun],         # past runs for same repo/author/branch
        ci_config: Optional[Dict] = None,   # parsed .yml config if available
    ) -> Dict[str, float]:
        """
        Main method. Returns a flat dict of numeric features.
        ALL values must be numeric (int or float). No strings.
        Missing values use sensible defaults.
        """
        features = {}

        # ── A) Code change features
        features.update(self._code_features(run))

        # ── B) Author historical features
        features.update(self._author_features(run, history))

        # ── C) Branch historical features
        features.update(self._branch_features(run, history))

        # ── D) Pipeline config features
        features.update(self._config_features(run, ci_config))

        # ── E) Temporal features
        features.update(self._temporal_features(run))

        # ── F) Commit message NLP features
        features.update(self._commit_message_features(run))

        return features

    # ────────────────────────────────────────
    # A) CODE FEATURES
    # ────────────────────────────────────────
    def _code_features(self, run: PipelineRun) -> Dict[str, float]:
        """
        What was changed in this commit?
        Large changes, no test changes, touching CI config = riskier.
        """
        files = run.files_changed

        # Count file types
        test_files    = [f for f in files if self._is_test_file(f)]
        config_files  = [f for f in files if self._is_ci_config(f)]
        infra_files   = [f for f in files if self._is_infra_file(f)]
        src_files     = [f for f in files if self._is_source_file(f)]

        total_files = len(files) if files else 0
        lines_total = run.lines_added + run.lines_deleted

        return {
            # Counts
            "files_changed_count":   float(total_files),
            "lines_added":           float(run.lines_added),
            "lines_deleted":         float(run.lines_deleted),
            "lines_changed_total":   float(lines_total),

            # File type indicators (0 or 1, or ratio)
            "has_test_changes":      float(len(test_files) > 0),
            "test_file_ratio":       len(test_files) / max(total_files, 1),
            "has_ci_config_change":  float(len(config_files) > 0),
            "has_infra_change":      float(len(infra_files) > 0),
            "source_file_count":     float(len(src_files)),

            # Risky signals
            "large_change_flag":     float(lines_total > 500),   # big changes = risky
            "no_test_with_src":      float(
                len(src_files) > 0 and len(test_files) == 0
            ),  # code changed but no tests = risky

            # Log-scale features (handle large numbers gracefully)
            "log_lines_changed":     math.log1p(lines_total),
            "log_files_changed":     math.log1p(total_files),
        }

    # ────────────────────────────────────────
    # B) AUTHOR FEATURES
    # ────────────────────────────────────────
    def _author_features(
        self, run: PipelineRun, history: List[PipelineRun]
    ) -> Dict[str, float]:
        """
        Historical track record of this specific developer.
        An author with 40% fail rate is a stronger predictor than file counts.
        """
        # Filter history to this author
        author_runs = [
            r for r in history
            if r.author_email == run.author_email
            and r.status in (PipelineStatus.SUCCESS, PipelineStatus.FAILED)
        ]

        if not author_runs:
            # No history → use neutral defaults
            return {
                "author_total_runs":       0.0,
                "author_fail_rate_all":    0.3,  # prior: 30% default
                "author_fail_rate_30d":    0.3,
                "author_fail_rate_7d":     0.3,
                "author_consecutive_fails":0.0,
                "author_avg_duration":     300.0,
            }

        def fail_rate(runs):
            if not runs: return 0.3
            failed = sum(1 for r in runs if r.status == PipelineStatus.FAILED)
            return failed / len(runs)

        now = datetime.utcnow()

        runs_30d = [
            r for r in author_runs
            if (now - r.triggered_at).days <= 30
        ]
        runs_7d = [
            r for r in author_runs
            if (now - r.triggered_at).days <= 7
        ]

        # Consecutive failures (most recent streak)
        sorted_runs = sorted(author_runs, key=lambda r: r.triggered_at, reverse=True)
        consecutive = 0
        for r in sorted_runs:
            if r.status == PipelineStatus.FAILED:
                consecutive += 1
            else:
                break

        # Average duration
        durations = [r.duration_seconds for r in author_runs if r.duration_seconds]
        avg_dur   = sum(durations) / len(durations) if durations else 300.0

        return {
            "author_total_runs":        float(len(author_runs)),
            "author_fail_rate_all":     fail_rate(author_runs),
            "author_fail_rate_30d":     fail_rate(runs_30d),
            "author_fail_rate_7d":      fail_rate(runs_7d),
            "author_consecutive_fails": float(consecutive),
            "author_avg_duration":      avg_dur,
        }

    # ────────────────────────────────────────
    # C) BRANCH FEATURES
    # ────────────────────────────────────────
    def _branch_features(
        self, run: PipelineRun, history: List[PipelineRun]
    ) -> Dict[str, float]:
        """
        History of this specific branch.
        Feature branches tend to be less stable than main/develop.
        """
        branch_runs = [
            r for r in history
            if r.branch == run.branch
            and r.status in (PipelineStatus.SUCCESS, PipelineStatus.FAILED)
        ]

        def fail_rate(runs):
            if not runs: return 0.25
            return sum(1 for r in runs if r.status == PipelineStatus.FAILED) / len(runs)

        now = datetime.utcnow()
        runs_7d = [r for r in branch_runs if (now - r.triggered_at).days <= 7]

        # Days since last failure on this branch
        failed_runs = sorted(
            [r for r in branch_runs if r.status == PipelineStatus.FAILED],
            key=lambda r: r.triggered_at, reverse=True
        )
        days_since_fail = (
            (now - failed_runs[0].triggered_at).days if failed_runs else 99
        )

        # Branch type (main/master/develop = stable, feature/* = risky)
        branch = run.branch.lower()
        is_main    = float(branch in ("main", "master", "develop", "dev"))
        is_release = float(branch.startswith("release/") or branch.startswith("hotfix/"))
        is_feature = float(branch.startswith("feature/") or branch.startswith("feat/"))

        return {
            "branch_fail_rate_all":    fail_rate(branch_runs),
            "branch_fail_rate_7d":     fail_rate(runs_7d),
            "branch_total_runs":       float(len(branch_runs)),
            "days_since_branch_fail":  float(min(days_since_fail, 99)),
            "branch_is_main":          is_main,
            "branch_is_release":       is_release,
            "branch_is_feature":       is_feature,
        }

    # ────────────────────────────────────────
    # D) CI CONFIG FEATURES
    # ────────────────────────────────────────
    def _config_features(
        self, run: PipelineRun, ci_config: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Features derived from the CI config file (.github/workflows/*.yml,
        .gitlab-ci.yml, Jenkinsfile).
        Parsed separately (see config_parser.py) and passed in here.
        """
        if not ci_config:
            return {
                "num_jobs":           1.0,
                "cache_enabled":      0.0,
                "parallel_enabled":   0.0,
                "has_test_stage":     0.0,
                "has_build_stage":    0.0,
                "has_deploy_stage":   0.0,
                "timeout_minutes":    60.0,
            }

        return {
            "num_jobs":           float(ci_config.get("num_jobs", 1)),
            "cache_enabled":      float(ci_config.get("cache_enabled", False)),
            "parallel_enabled":   float(ci_config.get("parallel_enabled", False)),
            "has_test_stage":     float(ci_config.get("has_test_stage", False)),
            "has_build_stage":    float(ci_config.get("has_build_stage", False)),
            "has_deploy_stage":   float(ci_config.get("has_deploy_stage", False)),
            "timeout_minutes":    float(ci_config.get("timeout_minutes", 60)),
        }

    # ────────────────────────────────────────
    # E) TEMPORAL FEATURES
    # ────────────────────────────────────────
    def _temporal_features(self, run: PipelineRun) -> Dict[str, float]:
        """
        When was this pipeline triggered?
        'Friday afternoon' is a real signal — devs rush before weekend.
        End of day pushes are historically riskier.
        """
        dt = run.triggered_at

        # Hour of day (0–23)
        hour = dt.hour
        # Day of week (0=Mon, 6=Sun)
        dow = dt.weekday()

        # Is it end-of-day (16:00–18:00)?
        is_eod = float(16 <= hour <= 18)
        # Is it Friday?
        is_friday = float(dow == 4)
        # Is it weekend?
        is_weekend = float(dow >= 5)
        # Is it late night (22:00–06:00)?
        is_late_night = float(hour >= 22 or hour <= 6)

        # "Danger zone": Friday afternoon
        is_friday_eod = float(is_friday and is_eod)

        return {
            "trigger_hour":       float(hour),
            "trigger_day_of_week":float(dow),
            "is_end_of_day":      is_eod,
            "is_friday":          is_friday,
            "is_weekend":         is_weekend,
            "is_late_night":      is_late_night,
            "is_friday_eod":      is_friday_eod,
        }

    # ────────────────────────────────────────
    # F) COMMIT MESSAGE NLP FEATURES
    # ────────────────────────────────────────
    def _commit_message_features(self, run: PipelineRun) -> Dict[str, float]:
        """
        Simple NLP features from commit message text.
        (Deep NLP like BERT comes later in nlp/classifier.py)
        Here we use lightweight signals.
        """
        msg = run.commit_message.lower() if run.commit_message else ""

        # Length signals
        msg_length    = len(msg)
        word_count    = len(msg.split())

        # Risky keyword patterns
        risky_keywords = [
            "fix", "hotfix", "urgent", "critical", "emergency",
            "hack", "temp", "wip", "quick", "dirty", "revert",
            "rollback", "breaking", "force", "patch"
        ]
        risk_keyword_count = sum(1 for kw in risky_keywords if kw in msg)

        # Positive signals
        positive_keywords = ["test", "docs", "refactor", "cleanup", "style"]
        positive_count = sum(1 for kw in positive_keywords if kw in msg)

        # Has ticket number (e.g. JIRA-123, #42) — structured = disciplined team
        has_ticket = float(bool(re.search(r'[A-Z]+-\d+|#\d+', run.commit_message or "")))

        # Conventional commit format (feat:, fix:, chore:) — good practice
        has_conventional = float(bool(
            re.match(r'^(feat|fix|chore|docs|style|refactor|test|build|ci)(\(.+\))?!?:', msg)
        ))

        # Very short message = probably rushed
        is_short_message = float(word_count <= 3)

        return {
            "commit_msg_length":        float(msg_length),
            "commit_msg_word_count":    float(word_count),
            "commit_risk_keyword_count":float(risk_keyword_count),
            "commit_positive_keywords": float(positive_count),
            "commit_has_ticket":        has_ticket,
            "commit_has_conventional":  has_conventional,
            "commit_is_short_msg":      is_short_message,
        }

    # ────────────────────────────────────────
    # HELPERS — file type classification
    # ────────────────────────────────────────
    def _is_test_file(self, path: str) -> bool:
        p = path.lower()
        return (
            "/test" in p or "/tests" in p or "/spec" in p
            or p.endswith("_test.py") or p.endswith(".test.js")
            or p.endswith(".spec.ts") or "test_" in p
        )

    def _is_ci_config(self, path: str) -> bool:
        p = path.lower()
        return (
            ".github/workflows" in p
            or ".gitlab-ci.yml" in p
            or "jenkinsfile" in p.split("/")[-1].lower()
            or "ci.yml" in p or "ci.yaml" in p
        )

    def _is_infra_file(self, path: str) -> bool:
        p = path.lower()
        return (
            "dockerfile" in p or ".tf" in p
            or "docker-compose" in p or "kubernetes" in p
            or "helm" in p or "k8s" in p
        )

    def _is_source_file(self, path: str) -> bool:
        exts = (".py", ".js", ".ts", ".java", ".go", ".rb", ".cs", ".cpp", ".rs")
        return any(path.lower().endswith(e) for e in exts)