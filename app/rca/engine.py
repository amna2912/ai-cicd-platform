

import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from ..nlp.classifier import KeywordLogClassifier, BERTLogClassifier, LogPreprocessor

@dataclass
class RCAReport:
    """Full root cause analysis report for one pipeline failure."""
    pipeline_run_id:      str
    root_cause_category:  str          # from ErrorClassifier
    confidence:           float
    error_snippet:        str          # the relevant log lines
    evidence:             List[str]    # specific evidence lines
    similar_incidents:    List[str]    # IDs of similar past failures
    shap_explanation:     Dict         # from SHAP (which features drove failure)
    recommendation:       str          # what to do to fix it
    llm_summary:          str          # human-readable paragraph
    pattern_id:           Optional[str] = None  # cluster ID for this error pattern



class IncidentClusterer:
    

    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.vectorizer  = None
        self.kmeans      = None
        self.cluster_labels: Dict[int, str] = {}  

    def fit(self, error_snippets: List[str]):
        """
        Fit the clustering model on historical error snippets.
        Call this periodically (e.g. weekly) to update clusters.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        print(f"[CLUSTER] Fitting on {len(error_snippets)} error snippets...")

       
        self.vectorizer = TfidfVectorizer(
            max_features    = 5000,
            ngram_range     = (1, 2),      # unigrams + bigrams
            analyzer        = "word",
            min_df          = 2,           # ignore very rare terms
            sublinear_tf    = True,        # log-scale TF
            strip_accents   = "unicode",
        )

        X = self.vectorizer.fit_transform(error_snippets)
        X = normalize(X)   # L2 normalize for cosine similarity

        actual_k = min(self.n_clusters, len(error_snippets))
        self.kmeans = KMeans(
            n_clusters  = actual_k,
            random_state= 42,
            n_init      = 10,
            max_iter    = 300,
        )
        self.kmeans.fit(X)

        print(f"[CLUSTER] Found {actual_k} clusters")
        return self

    def predict(self, error_snippet: str) -> Tuple[int, float]:
        """
        Assign a new error snippet to a cluster.
        Returns: (cluster_id, distance_to_centroid)
        """
        if not self.vectorizer or not self.kmeans:
            return -1, 1.0

        from sklearn.preprocessing import normalize

        X = self.vectorizer.transform([error_snippet])
        X = normalize(X)

        cluster_id = int(self.kmeans.predict(X)[0])

        centroid    = self.kmeans.cluster_centers_[cluster_id]
        distance    = float(np.linalg.norm(X.toarray() - centroid))

        return cluster_id, distance

    def find_similar(
        self,
        error_snippet: str,
        all_snippets: List[str],
        all_ids: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find the top-K most similar past incidents to a new error.
        Returns list of {id, similarity_score}.

        This uses cosine similarity on TF-IDF vectors.
        """
        if not self.vectorizer:
            return []

        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize

        query = normalize(self.vectorizer.transform([error_snippet]))
        corpus = normalize(self.vectorizer.transform(all_snippets))

        similarities = cosine_similarity(query, corpus)[0]

        ranked = sorted(
            zip(all_ids, similarities),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {"id": pid, "similarity": round(float(sim), 4)}
            for pid, sim in ranked[:top_k]
            if sim > 0.3   # minimum similarity threshold
        ]


CATEGORY_RECOMMENDATIONS = {
    "OOM": [
        "Increase container memory limit in your CI config (e.g. memory: 4Gi)",
        "Add JVM heap flags: -Xmx2g -Xms512m for Java builds",
        "Split the job into smaller parallel jobs",
        "Use streaming/chunked processing for large datasets",
        "Check for memory leaks in test setup/teardown",
    ],
    "DEP_MISSING": [
        "Run 'npm install' / 'pip install -r requirements.txt' before the build step",
        "Check if the package is in dependencies (not devDependencies) if used in prod",
        "Verify the package version exists: check registry.npmjs.org or pypi.org",
        "Add a cache step for node_modules / .pip to speed up future runs",
        "Pin dependency versions to avoid 'latest' resolution failures",
    ],
    "TEST_FAILURE": [
        "Review the failing test output to identify the specific assertion",
        "Check if the failure is environment-specific (database, external API mock)",
        "Run the test locally: pytest tests/test_foo.py -v",
        "If flaky, add retry logic or isolate the test",
        "Check if recent code changes broke the tested behavior",
    ],
    "BUILD_ERROR": [
        "Read the full compiler error — the first error is usually the root cause",
        "Check for syntax errors in recently changed files",
        "Verify all imported modules/types are correctly exported",
        "Run the build locally before pushing: npm run build / mvn compile",
        "Check if a dependency update changed an API you depend on",
    ],
    "NETWORK": [
        "Add retry logic with exponential backoff for network calls",
        "Check if external service (registry, artifact store) is available",
        "Verify DNS resolution works in the CI environment",
        "Use internal mirrors for package registries if on private network",
        "Add network timeout configuration to avoid indefinite hangs",
    ],
    "AUTH": [
        "Verify CI secrets/credentials are set in pipeline settings",
        "Check token expiry date — rotate if expired",
        "Ensure the service account has the required permissions",
        "Use environment-specific credentials (don't share staging/prod tokens)",
        "Check SSH key fingerprint if using git over SSH",
    ],
    "TIMEOUT": [
        "Increase job timeout: timeout: 30m in your CI config",
        "Enable parallelism to split long-running test suites",
        "Profile which step is slow and optimize it (caching, fewer tests)",
        "Add test splitting: pytest --splits 4 --group 1",
        "Check for hanging processes (infinite loops, waiting for input)",
    ],
    "FLAKY": [
        "Add automatic retry: retry: 2 in your CI config",
        "Isolate the flaky test and add a skip mark until fixed",
        "Check for shared state between tests (use fixtures for cleanup)",
        "Mock external service calls to remove network dependency",
        "Add sleep/wait for async operations instead of fixed delays",
    ],
    "UNKNOWN": [
        "Review the full log output starting from the first ERROR line",
        "Try running the pipeline locally to reproduce",
        "Check for recent changes to CI configuration",
        "Compare with the last successful pipeline run",
    ],
}


class RCAEngine:
    """
    Orchestrates the full Root Cause Analysis flow:
      1. Classify the error (keyword or BERT)
      2. Find similar past incidents
      3. Get SHAP explanation from the prediction model
      4. Generate recommendation
      5. Call LLM for human-readable summary
    """

    def __init__(
        self,
        use_bert: bool = False,
        llm_client = None,   
    ):
        self.preprocessor = LogPreprocessor()
        self.classifier   = (BERTLogClassifier() if use_bert
                             else KeywordLogClassifier())
        self.clusterer    = IncidentClusterer()
        self.llm_client   = llm_client

        if use_bert:
            self.classifier.load()

    def analyze(
        self,
        pipeline_run_id: str,
        raw_log: str,
        shap_explanation: Optional[Dict] = None,
        similar_incident_pool: Optional[List[Dict]] = None,  
    ) -> RCAReport:
        """
        Full RCA for one failed pipeline run.
        """
        clean_log = self.preprocessor.preprocess(raw_log)
        error_window = self.preprocessor.extract_error_window(clean_log)

        classification = self.classifier.classify(error_window)

        evidence = self._extract_evidence(clean_log, classification.category)

        similar = []
        if similar_incident_pool:
            snippets = [i["log_snippet"] for i in similar_incident_pool]
            ids      = [i["id"] for i in similar_incident_pool]
            try:
                similar = self.clusterer.find_similar(error_window, snippets, ids, top_k=3)
            except Exception:
                pass

        recs = CATEGORY_RECOMMENDATIONS.get(classification.category, CATEGORY_RECOMMENDATIONS["UNKNOWN"])
        primary_recommendation = recs[0]   

        llm_summary = self._generate_llm_summary(
            pipeline_run_id,
            classification.category,
            classification.error_snippet,
            shap_explanation,
            primary_recommendation,
        )

        return RCAReport(
            pipeline_run_id     = pipeline_run_id,
            root_cause_category = classification.category,
            confidence          = classification.confidence,
            error_snippet       = classification.error_snippet,
            evidence            = evidence,
            similar_incidents   = [s["id"] for s in similar],
            shap_explanation    = shap_explanation or {},
            recommendation      = primary_recommendation,
            llm_summary         = llm_summary,
        )

    def _extract_evidence(self, log: str, category: str) -> List[str]:
        """Extract the specific lines that are evidence for this category."""
        from app.nlp.classifier import CATEGORY_PATTERNS
        import re

        lines   = log.split("\n")
        evidence = []
        patterns = CATEGORY_PATTERNS.get(category, [])

        for line in lines:
            for pattern in patterns:
                if re.search(pattern, line.lower()):
                    cleaned = line.strip()
                    if cleaned and cleaned not in evidence:
                        evidence.append(cleaned)
                    break

        return evidence[:10]   # max 10 evidence lines

    def _generate_llm_summary(
        self,
        run_id: str,
        category: str,
        error_snippet: str,
        shap_explanation: Optional[Dict],
        recommendation: str,
    ) -> str:
        """
        Call an LLM (Claude/GPT) to generate a human-readable RCA paragraph.
        If no LLM client is configured, return a template-based summary.
        """
        if not self.llm_client:
            # Template-based fallback (no LLM required)
            return (
                f"Pipeline {run_id} failed due to a {category.replace('_', ' ').lower()} error. "
                f"The error was detected in: \"{error_snippet[:200]}\". "
                f"Recommended action: {recommendation}"
            )

        # ── LLM prompt
        shap_text = ""
        if shap_explanation and shap_explanation.get("top_factors"):
            factors = shap_explanation["top_factors"][:3]
            shap_text = "Contributing risk factors: " + ", ".join(
                f"{f['feature']} = {f['value']}" for f in factors
            )

        prompt = f"""You are a DevOps expert analyzing a CI/CD pipeline failure.

Pipeline ID: {run_id}
Error category: {category}
Error snippet from logs:
---
{error_snippet[:500]}
---
{shap_text}

Write a concise 2-3 sentence explanation of:
1. What went wrong and why
2. The most likely root cause
3. The specific action to fix it

Be concrete and actionable. Speak directly to the developer."""

        try:
            # Works with Anthropic Claude
            response = self.llm_client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 300,
                messages   = [{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"[LLM unavailable: {e}] {recommendation}"