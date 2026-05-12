"""
STEP 5 — NLP ERROR CLASSIFIER (BERT fine-tuning)
==================================================
This module takes RAW LOG TEXT from a failed pipeline
and classifies the error into a category.

WHY BERT AND NOT REGEX?
  Regex: matches "OutOfMemoryError" → "OOM"
  BERT:  understands "the process was killed because it consumed too much heap"
         → also "OOM" — even without the exact keyword.
  BERT generalizes. Regex breaks on new error patterns.

ERROR CATEGORIES WE CLASSIFY:
  - OOM           : out of memory (heap, RAM, container limits)
  - DEP_MISSING   : dependency not found, import error, package missing
  - TEST_FAILURE  : unit/integration test failed
  - BUILD_ERROR   : compile error, syntax error, type error
  - NETWORK       : timeout, connection refused, DNS failure
  - AUTH          : permission denied, credentials, token expired
  - TIMEOUT       : job exceeded time limit
  - FLAKY         : intermittent failure (passes on retry)
  - UNKNOWN       : cannot classify

FINE-TUNING WORKFLOW:
  1. Collect ~500-2000 labeled log snippets (label = error category)
  2. Fine-tune bert-base-uncased on them (takes ~20 min on GPU)
  3. Save the fine-tuned model
  4. Use it for inference in real-time (< 100ms)

For the MVP: use the keyword-based classifier first,
then replace it with BERT once you have labeled data.
"""

import re
import json
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────
# ERROR CATEGORY TAXONOMY
# ─────────────────────────────────────────────

ERROR_CATEGORIES = {
    "OOM":          "Out of Memory",
    "DEP_MISSING":  "Missing Dependency",
    "TEST_FAILURE": "Test Failure",
    "BUILD_ERROR":  "Build / Compile Error",
    "NETWORK":      "Network / Connectivity",
    "AUTH":         "Authentication / Authorization",
    "TIMEOUT":      "Timeout",
    "FLAKY":        "Flaky / Intermittent",
    "UNKNOWN":      "Unknown",
}

# Keyword patterns per category (used for labeling + as baseline)
CATEGORY_PATTERNS = {
    "OOM": [
        r"out of memory", r"outofmemory", r"heap space", r"java\.lang\.outofmemoryerror",
        r"cannot allocate memory", r"killed.*memory", r"oom killer",
        r"memory limit exceeded", r"container.*memory", r"segmentation fault",
    ],
    "DEP_MISSING": [
        r"modulenot found", r"cannot find module", r"no module named",
        r"importerror", r"package .* not found", r"could not find artifact",
        r"resolution failed", r"dependency.*not.*found", r"missing.*package",
        r"npm err.*404", r"pip.*no.*distribution",
    ],
    "TEST_FAILURE": [
        r"\d+ (test[s]? )?(failed|failure)", r"assertion.*failed",
        r"assertionerror", r"test.*error.*:", r"expected.*but.*got",
        r"junit.*failures", r"pytest.*failed", r"rspec.*failure",
        r"mocha.*failing", r"failing tests",
    ],
    "BUILD_ERROR": [
        r"syntaxerror", r"compileerror", r"typeerror.*at", r"cannot find symbol",
        r"build failed", r"compilation failed", r"undefined.*symbol",
        r"error: expected", r"gradle build failed", r"maven.*build.*failure",
        r"webpack.*error", r"tsc.*error",
    ],
    "NETWORK": [
        r"connection refused", r"connection timed out", r"could not resolve host",
        r"dns resolution failed", r"network.*unreachable", r"econnrefused",
        r"enotfound", r"socket.*timeout", r"max retries exceeded",
        r"ssl.*error", r"certificate.*error",
    ],
    "AUTH": [
        r"permission denied", r"access denied", r"unauthorized",
        r"authentication failed", r"invalid.*token", r"token.*expired",
        r"credentials.*not.*found", r"forbidden.*403", r"401.*unauthorized",
        r"ssh.*permission", r"gpg.*key",
    ],
    "TIMEOUT": [
        r"job.*timed out", r"execution.*timed out", r"deadline exceeded",
        r"timeout after \d+", r"pipeline.*timeout", r"step.*exceeded.*limit",
        r"signal: killed", r"process.*timed out",
    ],
    "FLAKY": [
        r"retry \d+", r"retrying.*attempt", r"intermittent",
        r"flaky.*test", r"random.*failure", r"non-deterministic",
    ],
}



@dataclass
class ClassificationResult:
    category: str
    category_label: str
    confidence: float
    matched_patterns: List[str]
    error_snippet: str      # the relevant log line(s)


class KeywordLogClassifier:
    """
    Rule-based classifier using regex patterns.
    Use this FIRST before you have labeled data for BERT fine-tuning.
    It's your baseline — BERT must beat this.

    Accuracy expectation: ~65-75% on common errors.
    BERT fine-tuned: ~88-92%.
    """

    def classify(self, log_text: str) -> ClassificationResult:
        """
        Classify a log text into an error category.
        Returns the most likely category + confidence + which patterns matched.
        """
        log_lower = log_text.lower()
        scores: Dict[str, int] = {}
        matches: Dict[str, List[str]] = {}

        for category, patterns in CATEGORY_PATTERNS.items():
            count = 0
            matched = []
            for pattern in patterns:
                if re.search(pattern, log_lower):
                    count += 1
                    matched.append(pattern)
            if count > 0:
                scores[category] = count
                matches[category] = matched

        if not scores:
            return ClassificationResult(
                category="UNKNOWN",
                category_label=ERROR_CATEGORIES["UNKNOWN"],
                confidence=0.0,
                matched_patterns=[],
                error_snippet=self._extract_error_snippet(log_text),
            )

        # Pick category with most pattern matches
        best_category = max(scores, key=scores.get)
        total_patterns = len(CATEGORY_PATTERNS[best_category])
        confidence = min(scores[best_category] / total_patterns, 1.0)

        return ClassificationResult(
            category=best_category,
            category_label=ERROR_CATEGORIES[best_category],
            confidence=round(confidence, 3),
            matched_patterns=matches[best_category],
            error_snippet=self._extract_error_snippet(log_text),
        )

    def _extract_error_snippet(self, log_text: str, max_lines: int = 5) -> str:
        """
        Extract the most relevant error lines from a long log.
        Look for lines with ERROR, FATAL, Exception, Error:, etc.
        """
        lines = log_text.split("\n")
        error_lines = [
            line.strip() for line in lines
            if re.search(r'\b(error|exception|fatal|fail|killed|denied)\b', line, re.IGNORECASE)
            and len(line.strip()) > 10
        ]
        if error_lines:
            return "\n".join(error_lines[:max_lines])

        # Fallback: last N non-empty lines
        non_empty = [l.strip() for l in lines if l.strip()]
        return "\n".join(non_empty[-max_lines:]) if non_empty else ""

    def label_log(self, log_text: str) -> Tuple[str, float]:
        """
        Convenience method: returns (category, confidence) tuple.
        Used to auto-label your training dataset for BERT fine-tuning.
        """
        result = self.classify(log_text)
        return result.category, result.confidence


# ─────────────────────────────────────────────
# BERT-BASED CLASSIFIER (Phase 2 — after labeling)
# ─────────────────────────────────────────────

class BERTLogClassifier:
    """
    Fine-tuned BERT classifier for log error classification.

    TRAINING STEPS (do this once offline):
      1. Collect logs from your database
      2. Auto-label with KeywordLogClassifier (for speed)
      3. Manually review and correct ~200 samples
      4. Fine-tune BERT using the training script below
      5. Save and load here

    This class loads the fine-tuned model and runs inference.
    """

    def __init__(self, model_path: str = "models/bert_classifier"):
        self.model_path = model_path
        self.model      = None
        self.tokenizer  = None
        self.label_map  = None
        self._loaded    = False

    def load(self):
        """Load the fine-tuned model from disk."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()

            # Load label map
            label_map_path = os.path.join(self.model_path, "label_map.json")
            with open(label_map_path) as f:
                self.label_map = json.load(f)

            self._loaded = True
            print(f"[BERT] Model loaded from {self.model_path}")
        except Exception as e:
            print(f"[BERT] Could not load model: {e}. Falling back to keyword classifier.")
            self._loaded = False

    def classify(self, log_text: str, max_length: int = 512) -> ClassificationResult:
        """
        Run BERT inference on a log snippet.
        We truncate to 512 tokens (BERT limit).
        For long logs: take the LAST 512 tokens (errors are usually at the end).
        """
        if not self._loaded:
            # Fallback to keyword classifier
            return KeywordLogClassifier().classify(log_text)

        import torch
        import torch.nn.functional as F

        # Take last 512 tokens worth of text (errors are at the end of logs)
        truncated_log = log_text[-2000:]   # roughly 512 tokens

        inputs = self.tokenizer(
            truncated_log,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = F.softmax(outputs.logits, dim=-1)
            pred_id = probs.argmax().item()
            confidence = probs[0][pred_id].item()

        category = self.label_map.get(str(pred_id), "UNKNOWN")

        return ClassificationResult(
            category=category,
            category_label=ERROR_CATEGORIES.get(category, category),
            confidence=round(confidence, 3),
            matched_patterns=[],
            error_snippet=KeywordLogClassifier()._extract_error_snippet(log_text),
        )


# ─────────────────────────────────────────────
# BERT FINE-TUNING SCRIPT
# ─────────────────────────────────────────────

BERT_TRAINING_CODE = '''
"""
Run this script ONCE to fine-tune BERT on your labeled log data.
Prerequisites:
    pip install transformers datasets torch scikit-learn

Data format (CSV):
    log_snippet,label
    "Error: Cannot find module 'express'",DEP_MISSING
    "OutOfMemoryError: Java heap space",OOM
    "1 test failed: AssertionError",TEST_FAILURE
    ...

Usage:
    python nlp/train_bert.py --data data/labeled_logs.csv --output models/bert_classifier
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json, os, argparse


CATEGORIES = ["OOM", "DEP_MISSING", "TEST_FAILURE", "BUILD_ERROR",
              "NETWORK", "AUTH", "TIMEOUT", "FLAKY", "UNKNOWN"]
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for cat, i in LABEL2ID.items()}


class LogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = labels

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_bert(data_path, output_dir):
    df = pd.read_csv(data_path)
    df = df[df["label"].isin(CATEGORIES)]
    print(f"Loaded {len(df)} labeled samples")
    print(df["label"].value_counts())

    texts  = df["log_snippet"].tolist()
    labels = [LABEL2ID[l] for l in df["label"].tolist()]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    MODEL_NAME = "bert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CATEGORIES),
        id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = LogDataset(X_train, y_train, tokenizer)
    val_ds   = LogDataset(X_val,   y_val,   tokenizer)

    training_args = TrainingArguments(
        output_dir             = output_dir,
        num_train_epochs       = 5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        learning_rate          = 2e-5,        # standard for BERT fine-tuning
        weight_decay           = 0.01,
        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "f1_macro",
        warmup_ratio           = 0.1,
        logging_steps          = 50,
        fp16                   = torch.cuda.is_available(),  # use GPU if available
    )

    trainer = Trainer(
        model          = model,
        args           = training_args,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        compute_metrics= compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label map for inference
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({str(v): k for k, v in LABEL2ID.items()}, f)

    print(f"\\nModel saved to: {output_dir}")
    results = trainer.evaluate()
    print(f"Final F1 (macro): {results['eval_f1_macro']:.4f}")
    print(f"Final Accuracy:   {results['eval_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--output", default="models/bert_classifier")
    args = parser.parse_args()
    train_bert(args.data, args.output)
'''

# Write BERT training script to file
def write_bert_training_script(output_path: str = "nlp/train_bert.py"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(BERT_TRAINING_CODE.strip())
    print(f"[BERT] Training script written to: {output_path}")


# ─────────────────────────────────────────────
# LOG PREPROCESSING
# ─────────────────────────────────────────────

class LogPreprocessor:
    """
    Clean and normalize raw CI logs before feeding to the classifier.
    Raw logs are noisy: timestamps, ANSI codes, duplicate lines.
    """

    def preprocess(self, raw_log: str) -> str:
        """Remove noise, keep signal."""
        lines = raw_log.split("\n")
        cleaned = []

        for line in lines:
            # Remove ANSI escape codes (terminal colors)
            line = re.sub(r'\x1B\[[0-9;]*[mGKH]', '', line)
            # Remove timestamps (2025-01-15T10:30:01.123Z)
            line = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*Z?\s*', '', line)
            # Remove log level prefixes [INFO], [DEBUG], etc.
            line = re.sub(r'^\s*\[(INFO|DEBUG|TRACE)\]\s*', '', line)
            # Trim
            line = line.strip()

            # Skip empty lines and pure separator lines
            if not line or re.match(r'^[-=*#]{5,}$', line):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    def extract_error_window(self, log: str, window: int = 50) -> str:
        """
        Find the first ERROR/FATAL line and return ±window lines around it.
        This gives BERT the most relevant context without 10,000 lines of noise.
        """
        lines = log.split("\n")
        for i, line in enumerate(lines):
            if re.search(r'\b(error|exception|fatal|failed)\b', line, re.IGNORECASE):
                start = max(0, i - 10)
                end   = min(len(lines), i + window)
                return "\n".join(lines[start:end])
        # No error found → return last 50 lines
        return "\n".join(lines[-window:])