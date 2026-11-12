"""Observable pre-treatment v/rho/e features and frozen text encoders.

The structured-only mode is an explicit baseline. It is not a fake text encoder.
Private checkpoints, arbitrary metadata and outcomes are not accepted inputs.
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

from .contracts import FeatureRecord, PublicContext, SourceMemory, canonical, digest, finite, read_json, truth, write_json
from .source import validate_evidence

DEFAULT_FACT_NAMES = frozenset({"phase", "prepared", "mechanism", "intent", "requires_preparation",
                               "task_family", "object_type", "receptacle_type", "object_count",
                               "goal_relation", "backed_up", "archive_available", "inventory",
                               "admissible_commands", "admissible_count", "location", "holding", "clean", "hot", "cool"})
FORBIDDEN_FACT_NAMES = frozenset({"actor_profile", "future_success_marker", "success", "terminal_success",
                                 "outcome", "targets", "label", "reward", "score", "environment_state",
                                 "actor_state", "rng", "expert_action", "stratum", "scenario", "metadata",
                                 "task_id", "checkpoint_id", "source_task_id"})


def _public_fact(value):
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        finite(value, "public fact")
        return True
    if isinstance(value, (tuple, list)):
        return all(not isinstance(v, (dict, tuple, list)) and _public_fact(v) for v in value)
    return False


def _reject_private_fields(value):
    if isinstance(value, dict):
        private = set(value) & FORBIDDEN_FACT_NAMES
        if private:
            raise ValueError(f"private/outcome fields cannot enter action/history features: {sorted(private)}")
        for child in value.values():
            _reject_private_fields(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_private_fields(child)


def _lookup(facts: dict, field: str):
    current: Any = facts
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _condition(condition: dict, context: PublicContext, allowed_fact_names):
    if not isinstance(condition, dict) or set(condition) - {"field", "equals", "not_equals", "in"} or "field" not in condition:
        return "?"
    field = condition["field"]
    if not isinstance(field, str) or field not in allowed_fact_names or field in FORBIDDEN_FACT_NAMES:
        return "?"
    if len(set(condition) & {"equals", "not_equals", "in"}) != 1:
        return "?"
    exists, value = _lookup(context.facts, field)
    if not exists:
        return "?"
    if "equals" in condition:
        return int(type(value) is type(condition["equals"]) and value == condition["equals"])
    if "not_equals" in condition:
        return int(not (type(value) is type(condition["not_equals"]) and value == condition["not_equals"]))
    if "in" in condition and isinstance(condition["in"], (list, tuple)):
        return int(any(type(value) is type(v) and value == v for v in condition["in"]))
    return "?"


def _operation(action: dict) -> str | None:
    if action.get("type") != "alfworld_text":
        return action.get("type") if isinstance(action.get("type"), str) else None
    text = " ".join(str(action.get("text", "")).lower().split())
    if not text:
        return None
    for prefix, operation in (("go to ", "go_to"), ("turn on ", "turn_on"), ("put ", "move"), ("move ", "move")):
        if text.startswith(prefix):
            return operation
    return text.split(" ", 1)[0]


def _matches_action(pattern: dict, action: dict) -> bool:
    """Only explicitly registered action structures can establish a match."""
    if not isinstance(pattern, dict) or not pattern.get("type"):
        raise ValueError("exposure action patterns need a declared type")
    exact = {k: v for k, v in pattern.items() if k != "command_regex"}
    if any(action.get(k) != value for k, value in exact.items()):
        return False
    if "command_regex" in pattern:
        return (isinstance(action.get("command"), str) and
                re.fullmatch(pattern["command_regex"], action["command"]) is not None)
    return True


def _exposure(context, action, memory, allowed_fact_names, rule=None):
    if rule is not None:
        allowed = {"trigger_actions", "safe_actions", "trigger_when", "protected_when"}
        if not isinstance(rule, dict) or set(rule) - allowed or not rule.get("trigger_actions"):
            raise ValueError("invalid registered exposure rule")
        if any(_matches_action(p, action) for p in rule.get("safe_actions", [])):
            return 0
        if not any(_matches_action(p, action) for p in rule["trigger_actions"]):
            return "?"
        trigger = [_condition(c, context, allowed_fact_names) for c in rule.get("trigger_when", [])]
        protected = [_condition(c, context, allowed_fact_names) for c in rule.get("protected_when", [])]
        if 0 in trigger or (protected and all(v == 1 for v in protected)):
            return 0
        if "?" in trigger or (protected and "?" in protected and 0 not in protected):
            return "?"
        return 1
    # This guard belongs to the explicitly typed development environment only.
    # Generic command equality does not establish mechanism exposure.
    fixture_contract = (memory.problem_action == {"type": "operate"}
                        and memory.correction == {"type": "prepare"}
                        and {"field": "requires_preparation", "equals": True} in memory.positive_conditions)
    if fixture_contract:
        if action in ({"type": "prepare"}, {"type": "finish"}):
            return 0
        if action == {"type": "operate"}:
            protected = context.facts.get("prepared")
            return int(not protected) if type(protected) is bool else "?"
    return "?"


class MechanismAssessor:
    """Versioned, preregistered action/state rules; no hidden-state access."""
    def __init__(self, rules: dict | None = None, *, allowed_fact_names=DEFAULT_FACT_NAMES):
        self.rules = copy.deepcopy(rules or {})
        self.allowed_fact_names = frozenset(allowed_fact_names)
        if self.allowed_fact_names & FORBIDDEN_FACT_NAMES:
            raise ValueError("private fields cannot be used by mechanism rules")
        self.version = "mechanism-assessor.v1:" + digest({"rules": self.rules, "facts": sorted(self.allowed_fact_names)})

    def __call__(self, context, action, memory):
        if self.version != "mechanism-assessor.v1:" + digest({"rules": self.rules, "facts": sorted(self.allowed_fact_names)}):
            raise ValueError("frozen exposure rules changed")
        rule = self.rules.get(memory.content_hash(), self.rules.get(memory.mechanism)) if memory else None
        return assess(context, action, memory, allowed_fact_names=self.allowed_fact_names, exposure_rule=rule)


class ObservationFactExtractor:
    """Facts supported by reviewed diagnostic actions and their public output.

    State facts expire at the next action by default. Optional preservation is
    restricted to explicitly reviewed actions, with all other actions clearing
    the fact. Command identity and zero exit status are evidence boundaries,
    not a proof that an actor cannot modify a diagnostic's implementation.
    """
    def __init__(self, rules: dict | None = None):
        self.rules = copy.deepcopy(rules or {})
        for field, rule in self.rules.items():
            if not isinstance(field, str) or not field or field in FORBIDDEN_FACT_NAMES or field == "phase":
                raise ValueError("invalid/private public fact field")
            allowed = {"source", "patterns", "diagnostic_actions", "diagnostic_basis", "persistence",
                       "preserve_actions", "persistence_basis", "kind"}
            if not isinstance(rule, dict) or set(rule) - allowed or not {"source", "patterns"} <= set(rule):
                raise ValueError("fact rules need explicit source and patterns")
            if rule["source"] not in ("observation", "instruction", "history") or not isinstance(rule["patterns"], list) or not rule["patterns"]:
                raise ValueError("invalid public fact source/patterns")
            if rule["source"] == "instruction":
                if rule.get("kind") != "task_static":
                    raise ValueError("instruction facts must explicitly declare kind=task_static")
                if any(k in rule for k in ("diagnostic_actions", "preserve_actions", "persistence")):
                    raise ValueError("task-static facts cannot declare execution state rules")
            else:
                if (not isinstance(rule.get("diagnostic_actions"), list) or not rule["diagnostic_actions"]
                        or not isinstance(rule.get("diagnostic_basis"), str) or not rule["diagnostic_basis"].strip()):
                    raise ValueError("state facts need reviewed diagnostic_actions and diagnostic_basis")
                persistence = rule.get("persistence", "until_next_action")
                if persistence not in ("until_next_action", "reviewed_actions"):
                    raise ValueError("unsupported fact persistence contract")
                if persistence == "reviewed_actions" and (not isinstance(rule.get("preserve_actions"), list)
                        or not rule["preserve_actions"] or not isinstance(rule.get("persistence_basis"), str)
                        or not rule["persistence_basis"].strip()):
                    raise ValueError("preserved facts need reviewed preserve_actions and persistence_basis")
                if persistence == "until_next_action" and rule.get("preserve_actions"):
                    raise ValueError("preserve_actions requires reviewed_actions persistence")
                if rule["source"] == "observation" and persistence != "until_next_action":
                    raise ValueError("current observation facts expire at the next action")
                for action in [*rule["diagnostic_actions"], *rule.get("preserve_actions", [])]:
                    if (not isinstance(action, dict) or set(action) != {"type", "command"}
                            or action["type"] != "shell" or not isinstance(action["command"], str)
                            or not action["command"].strip() or "\x00" in action["command"]):
                        raise ValueError("fact provenance requires exact shell actions, not action regexes")
            for pattern in rule["patterns"]:
                if not isinstance(pattern, dict) or set(pattern) != {"regex", "value"} or not _public_fact(pattern["value"]):
                    raise ValueError("fact patterns need a regex and public scalar/list value")
                re.compile(pattern["regex"])
        self.version = "public-observation-facts.v2-action-provenance:" + digest(self.rules)

    @staticmethod
    def _matched_value(text, patterns):
        matches = [(match.start(), index, pattern["value"])
                   for index, pattern in enumerate(patterns) for match in re.finditer(pattern["regex"], text)]
        return (True, copy.deepcopy(max(matches, key=lambda item: (item[0], item[1]))[2])) if matches else (False, None)

    @staticmethod
    def _diagnostic_output(event):
        try:
            output = json.loads(event.get("observation", ""))
        except (TypeError, ValueError):
            return None
        if (not isinstance(output, dict) or type(output.get("exit_code")) is not int or output["exit_code"] != 0
                or output.get("command_timed_out") or output.get("timed_out") or output.get("truncated")
                or not isinstance(output.get("stdout"), str)):
            return None
        return output["stdout"]

    def extract(self, context: PublicContext) -> dict:
        if self.version != "public-observation-facts.v2-action-provenance:" + digest(self.rules):
            raise ValueError("frozen public fact rules changed")
        facts = {}
        for field, rule in self.rules.items():
            if rule["source"] == "instruction":
                matched, value = self._matched_value(context.instruction, rule["patterns"])
                if matched:
                    facts[field] = value
                continue
            history = context.history if rule.get("persistence") == "reviewed_actions" else context.history[-1:]
            if rule["source"] == "observation" and (not history or history[-1].get("observation") != context.observation):
                continue
            for event in history:
                action = event.get("action")
                if any(action == candidate for candidate in rule["diagnostic_actions"]):
                    facts.pop(field, None)
                    output = self._diagnostic_output(event)
                    if output is not None:
                        matched, value = self._matched_value(output, rule["patterns"])
                        if matched:
                            facts[field] = value
                elif not any(action == candidate for candidate in rule.get("preserve_actions", [])):
                    facts.pop(field, None)
        return facts


def assess(context: PublicContext, draft_action: dict, memory: SourceMemory | None, *,
           allowed_fact_names=DEFAULT_FACT_NAMES, exposure_rule: dict | None = None) -> dict:
    """Separate evidence validity, mechanism applicability and action exposure."""
    if not isinstance(context, PublicContext) or not isinstance(draft_action, dict):
        raise TypeError("assess requires PublicContext and a typed draft action")
    if memory is None:
        return {"v": "?", "rho": "?", "e": "?", "v_confidence": 0.0,
                "rho_confidence": 0.0, "e_confidence": 0.0, "reason": "no_candidate"}
    evidence = validate_evidence(memory)
    allowed_fact_names = frozenset(allowed_fact_names) - FORBIDDEN_FACT_NAMES
    positive = [_condition(c, context, allowed_fact_names) for c in memory.positive_conditions]
    excluded = [_condition(c, context, allowed_fact_names) for c in memory.exclusion_conditions]
    if 0 in positive or 1 in excluded:
        rho = 0
    elif "?" in positive or "?" in excluded or not positive:
        rho = "?"
    else:
        rho = 1
    exposure = _exposure(context, draft_action, memory, allowed_fact_names, exposure_rule)
    return {"v": evidence["v"], "rho": rho, "e": exposure,
            "v_confidence": 0.0 if evidence["v"] == "?" else 1.0,
            "rho_confidence": 0.0 if rho == "?" else 1.0,
            "e_confidence": 0.0 if exposure == "?" else 1.0,
            "reason": evidence["reason"], "evidence_scope": evidence["scope"]}


class FrozenEncoder:
    """Versioned callable encoder with content-addressed persistent caching."""
    def __init__(self, encode_fn: Callable[[list[str]], list[list[float]]], *, version: str,
                 cache_dir: str | Path | None = None):
        if not version:
            raise ValueError("encoder version must identify weights and preprocessing")
        self.encode_fn, self.version = encode_fn, version
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, list[float]] = {}
        self.dimension: int | None = None
        self.last_usage = {"encoder_calls": 0, "cache_hits": 0, "wall_seconds": 0.0}

    def encode(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("encoder inputs must be text")
        started = time.perf_counter()
        keys = [digest({"encoder": self.version, "text": text}) for text in texts]
        missing, missing_texts = [], []
        for key, text in zip(keys, texts):
            if key not in self._cache and self.cache_dir and (self.cache_dir / f"{key}.json").exists():
                row = read_json(self.cache_dir / f"{key}.json")
                if row.get("encoder") != self.version or row.get("text_hash") != digest(text):
                    raise ValueError("encoder cache identity mismatch")
                self._cache[key] = self._validate(row["vector"])
            if key not in self._cache and key not in missing:
                missing.append(key)
                missing_texts.append(text)
        if missing:
            vectors = self.encode_fn(missing_texts)
            if len(vectors) != len(missing):
                raise ValueError("encoder result count mismatch")
            for key, text, vector in zip(missing, missing_texts, vectors):
                self._cache[key] = self._validate(vector)
                if self.cache_dir:
                    write_json(self.cache_dir / f"{key}.json", {"encoder": self.version,
                               "text_hash": digest(text), "vector": self._cache[key]})
        self.last_usage = {"encoder_calls": int(bool(missing)), "cache_hits": len(texts) - len(missing),
                           "wall_seconds": time.perf_counter() - started}
        return [copy.deepcopy(self._cache[key]) for key in keys]

    def _validate(self, vector) -> list[float]:
        vector = list(vector)
        if any(isinstance(x, bool) for x in vector):
            raise ValueError("embedding values cannot be boolean")
        vector = [finite(float(x), "embedding") for x in vector]
        if not vector:
            raise ValueError("empty embedding")
        if self.dimension is None:
            self.dimension = len(vector)
        if len(vector) != self.dimension:
            raise ValueError("encoder dimension changed")
        return vector


class SentenceTransformerEncoder(FrozenEncoder):
    """Actual frozen sentence-transformer; explicit revision is required."""
    def __init__(self, model_name: str, *, revision: str, device: str = "cpu",
                 max_sequence_length: int = 512, cache_dir: str | Path | None = None):
        if not model_name or not revision:
            raise ValueError("pin the encoder name/path and weight revision")
        if type(max_sequence_length) is not int or max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self.model.max_seq_length = max_sequence_length
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        version = digest({"encoder": model_name, "revision": revision, "normalize": True,
                          "max_sequence_length": max_sequence_length})
        super().__init__(lambda texts: self.model.encode(texts, normalize_embeddings=True,
                                                         convert_to_numpy=True).tolist(),
                         version=version, cache_dir=cache_dir)


class FeatureBuilder:
    """Explicit, stable numerical features; never exports raw arbitrary dicts."""
    def __init__(self, *, encoder: FrozenEncoder | None = None,
                 allowed_fact_names=DEFAULT_FACT_NAMES):
        self.encoder = encoder
        self.allowed_fact_names = frozenset(allowed_fact_names)
        if self.allowed_fact_names & FORBIDDEN_FACT_NAMES:
            raise ValueError("privileged/outcome facts cannot be allowlisted")
        self.version = "features.v2:" + digest({"encoder": getattr(encoder, "version", "structured-only"),
                                               "facts": sorted(self.allowed_fact_names),
                                               "structured_map": "tri-state-public-v2.2-full-memory"})[:20]
        self.last_usage: dict[str, float | None] = {}

    def build(self, context: PublicContext, draft_action: dict, memory: SourceMemory | None,
              checks: dict | None = None, *, remaining_calls: int | None = None,
              retrieval_score: float = 0.0, divergence: float | None = None) -> FeatureRecord:
        started = time.perf_counter()
        if not isinstance(context, PublicContext):
            raise TypeError("private/raw context is not accepted")
        unexpected = set(context.facts) - self.allowed_fact_names
        if unexpected:
            raise ValueError(f"unapproved public fact fields: {sorted(unexpected)}")
        if not all(_public_fact(value) for value in context.facts.values()):
            raise ValueError("public facts must use declared scalar or scalar-list values")
        if not isinstance(draft_action, dict):
            raise TypeError("draft must be a typed public action")
        _reject_private_fields(draft_action)
        for event in context.history:
            if not isinstance(event, dict) or set(event) - {"action", "observation"}:
                raise ValueError("history must contain only executed actions and public observations")
            _reject_private_fields(event.get("action"))
        checks = assess(context, draft_action, memory, allowed_fact_names=self.allowed_fact_names) if checks is None else copy.deepcopy(checks)
        unknown_check_fields = set(checks) - {"v", "rho", "e", "v_confidence", "rho_confidence", "e_confidence", "reason", "evidence_scope"}
        if unknown_check_fields:
            raise ValueError("unapproved assessment fields")
        names, values = [], []
        for key in ("v", "rho", "e"):
            value = truth(checks[key])
            names.extend((key, key + "_unknown", key + "_confidence"))
            confidence = finite(checks.get(key + "_confidence", 0.0 if value == "?" else 1.0), "confidence", minimum=0)
            if confidence > 1:
                raise ValueError("confidence outside [0,1]")
            values.extend((0.0 if value == "?" else float(value), float(value == "?"), confidence))
        for key, value in (("remaining_steps", context.remaining_steps),
                           ("remaining_calls", remaining_calls if remaining_calls is not None else context.remaining_steps),
                           ("history_length", len(context.history)), ("admissible_count", len(context.admissible_actions)),
                           ("retrieval_score", retrieval_score), ("divergence", 0.0 if divergence is None else divergence),
                           ("divergence_unknown", float(divergence is None))):
            names.append(key)
            values.append(finite(value, key, minimum=0) if key != "retrieval_score" else finite(value, key))
        for key in ("prepared", "requires_preparation", "backed_up", "archive_available"):
            value = context.facts.get(key)
            names.extend(("fact_" + key, "fact_" + key + "_unknown"))
            values.extend((float(value) if type(value) is bool else 0.0, float(type(value) is not bool)))
        if self.encoder is not None:
            # No IDs, private states or arbitrary evidence records go to encoder.
            evidence_summary = {} if memory is None else {k: memory.evidence.get(k) for k in
                                ("scope", "endpoint", "mean_difference", "lower_bound", "upper_bound")}
            blocks = [context.instruction + "\n" + context.observation + "\n" + canonical(context.history)
                      + "\n" + canonical(context.facts),
                      canonical(draft_action), "" if memory is None else
                      memory.mechanism + "\n" + canonical({"problem_action": memory.problem_action,
                          "correction": memory.correction, "positive_conditions": memory.positive_conditions,
                          "exclusion_conditions": memory.exclusion_conditions, "evidence": evidence_summary})]
            vectors = self.encoder.encode(blocks)
            for block, vector in enumerate(vectors):
                for i, value in enumerate(vector):
                    names.append(f"text_{block}_{i}")
                    values.append(value)
        self.last_usage = {"feature_calls": 1, "encoder_calls": 0 if self.encoder is None else self.encoder.last_usage["encoder_calls"],
                           "wall_seconds": time.perf_counter() - started,
                           "api_cost": 0.0 if self.encoder is None else None}
        return FeatureRecord(tuple(values), tuple(names), self.version)

    def __call__(self, context, draft_action, memory, **kwargs):
        return self.build(context, draft_action, memory, **kwargs)
