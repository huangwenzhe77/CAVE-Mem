"""Explicit component-matched baselines and an isolated external-results bridge.

External systems are never imported into the research kernel. Their results are
read from declared JSON exports; an adapter is not silently called a faithful
paper reproduction. No messages or API requests are sent by this module.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import digest, read_json, read_jsonl, write_json
from .learning import _number, eligible


def _check(row, key):
    if key in row.get("checks", {}):
        value = row["checks"][key]
        return value.get("value", "?") if isinstance(value, dict) else value
    values = dict(zip(row["feature_names"], row["x"]))
    aliases = {"v": ("validity", "v"), "rho": ("applicability", "rho"),
               "e": ("exposure", "e")}
    for alias in aliases.get(key, (key,)):
        if alias in values:
            return values[alias]
    return "?"


@dataclass(frozen=True)
class RulePolicy:
    """Component-matched decision policies; no fabricated published-system names."""
    name: str = "silent"
    relevance_threshold: float = 0.

    def __post_init__(self):
        allowed = {"silent", "always_remind", "relevance", "source_only", "applicability",
                   "exposure", "hard_vre"}
        if self.name not in allowed:
            raise ValueError(f"unknown rule baseline {self.name}; choose {sorted(allowed)}")
        _number(self.relevance_threshold, "relevance threshold")

    def predict_probability(self, rows):
        values = []
        for row in rows:
            v, rho, e = (_check(row, key) for key in ("v", "rho", "e"))
            if self.name == "silent":
                answer = False
            elif self.name == "always_remind":
                # Invalid-bank/noisy-memory experiments explicitly set eligible
                # according to their protocol; published-system behavior is separate.
                answer = row.get("has_candidate", True)
            elif self.name == "source_only":
                answer = v == 1
            elif self.name == "applicability":
                answer = v == 1 and rho == 1
            elif self.name == "exposure":
                answer = v == 1 and e == 1
            elif self.name == "hard_vre":
                answer = v == rho == e == 1
            else:
                score = row.get("retrieval_score")
                if score is None:
                    features = dict(zip(row["feature_names"], row["x"]))
                    score = features.get("retrieval_score")
                if score is None:
                    raise ValueError("relevance baseline requires a pre-treatment retrieval_score")
                answer = _number(score, "retrieval score") >= self.relevance_threshold
            values.append(float(answer))
        return np.asarray(values)


class SupervisedGate:
    """Threshold the same calibrated outcome encoder/heads used to warm start RL."""
    def __init__(self, outcome_model, risk_threshold=1.0, use_lower=True):
        if outcome_model.calibration is None:
            raise ValueError("supervised gate needs held-out calibration")
        if not 0 <= risk_threshold <= 1:
            raise ValueError("risk threshold must be in [0,1]")
        self.outcome_model = copy.deepcopy(outcome_model)
        self.risk_threshold, self.use_lower = risk_threshold, use_lower
        self.training_rows = outcome_model.training_rows

    def predict_probability(self, rows):
        pred = self.outcome_model.predict(rows)
        tau = pred["tau_lower"] if self.use_lower else pred["tau"]
        return np.asarray([eligible(row) for row in rows], dtype=float) * (tau > 0) * (pred["risk"] <= self.risk_threshold)


@dataclass(frozen=True)
class ExternalRunManifest:
    system: str
    implementation_kind: str
    repository: str
    commit: str
    configuration_hash: str
    actor_hash: str
    environment_hash: str
    task_set_hash: str
    budget_protocol_hash: str
    upstream_verification: str = "unverified_local_checkout"
    model_weights: str = "not_declared"
    notes: str = ""

    def __post_init__(self):
        if self.implementation_kind not in ("component_matched", "intact_local", "verified_original"):
            raise ValueError("declare component_matched, intact_local, or verified_original")
        for key in ("system", "repository", "commit", "configuration_hash", "actor_hash",
                    "environment_hash", "task_set_hash", "budget_protocol_hash"):
            if not getattr(self, key):
                raise ValueError(f"external baseline missing {key}")
        if self.implementation_kind == "verified_original" and self.upstream_verification == "unverified_local_checkout":
            raise ValueError("a local mirror is not automatically a verified original")


def import_external_results(manifest: ExternalRunManifest, path: str | Path, *, expected=None,
                            expected_task_ids=None):
    """Validate standardized task-level exports without launching the external repo.

    Required rows: task_id, seed, success, status, costs. Infrastructure errors
    are retained and excluded explicitly; unknown costs remain unknown.
    """
    path = Path(path)
    rows = read_jsonl(path) if path.suffix.lower() == ".jsonl" else read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("external export must contain task-level results")
    if expected_task_ids is not None and {row.get("task_id") for row in rows} != set(expected_task_ids):
        raise ValueError("external export does not cover exactly the registered task set")
    if expected:
        for key, value in expected.items():
            if getattr(manifest, key, None) != value:
                raise ValueError(f"external baseline mismatch: {key}")
    seen, valid, excluded = set(), [], []
    for row in rows:
        if not isinstance(row, dict) or not row.get("task_id") or type(row.get("seed")) is not int:
            raise ValueError("external result requires task_id and integer seed")
        identity = row["task_id"], row["seed"]
        if identity in seen:
            raise ValueError("duplicate external task/seed")
        seen.add(identity)
        if row.get("status") == "error":
            excluded.append({**row, "exclusion_reason": "external_infrastructure_error"})
            continue
        if row.get("status") not in ("terminal", "budget_exhausted", "invalid_action") or type(row.get("success")) is not bool:
            raise ValueError("declare valid fixed-budget external outcome status and boolean success")
        if row["status"] != "terminal" and row["success"]:
            raise ValueError("nonterminal/budget-exhausted outcome cannot claim success")
        costs = row.get("costs")
        if not isinstance(costs, dict) or not costs:
            raise ValueError("external result needs explicit measured or unknown costs")
        for key, value in costs.items():
            if value is not None and _number(value, key) < 0:
                raise ValueError("external costs cannot be negative")
        valid.append(row)
    return {"schema_version": "external-baseline.v1", "manifest": manifest.__dict__,
            "manifest_hash": digest(manifest), "source_hash": digest(rows), "results": valid,
            "excluded": excluded, "planned_tasks": len(rows), "valid_tasks": len(valid),
            "success_rate": sum(r["success"] for r in valid) / len(valid) if valid else None,
            "paired_regression": None, "comparison_scope": "task-level complete-system results",
            "note": "separate task exports do not establish matched-fork harm or equal component behavior"}


def write_external_template(path, *, system="MemRL"):
    """Write a reviewable exchange specification, not an executable runner stub."""
    write_json(path, {"schema_version": "external-baseline-template.v1", "system": system,
                     "required_manifest": list(ExternalRunManifest.__dataclass_fields__),
                     "result_fields": {"task_id": "dataset task identity", "seed": "integer",
                         "success": "boolean", "status": "terminal|budget_exhausted|invalid_action|error",
                         "costs": {"input_tokens": None, "output_tokens": None, "tool_calls": None,
                                   "wall_seconds": None}},
                     "scope": "run the pinned external repository in a separate environment/process, then export measured task outcomes"})
