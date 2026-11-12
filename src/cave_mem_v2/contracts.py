"""Shared data and numerical contracts; never import providers or environments."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

Truth = Literal[0, 1, "?"]


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def truth(value: Any) -> Truth:
    if value == "?" or type(value) is int and value in (0, 1):
        return value
    raise ValueError("truth value must be integer 0, integer 1, or '?' (not bool)")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class PublicContext:
    task_id: str
    instruction: str
    observation: str
    facts: dict[str, Any] = field(default_factory=dict)
    history: tuple[dict[str, Any], ...] = ()
    admissible_actions: tuple[str, ...] = ()
    remaining_steps: int = 20

    def __post_init__(self) -> None:
        if not self.task_id or not self.instruction:
            raise ValueError("task_id and instruction are required")
        if type(self.remaining_steps) is not int or self.remaining_steps < 0:
            raise ValueError("remaining_steps must be nonnegative integer")
        object.__setattr__(self, "facts", copy.deepcopy(self.facts))
        object.__setattr__(self, "history", tuple(copy.deepcopy(self.history)))
        object.__setattr__(self, "admissible_actions", tuple(self.admissible_actions))
        canonical(self)


@dataclass(frozen=True)
class FeatureRecord:
    values: tuple[float, ...]
    names: tuple[str, ...]
    version: str = "decision-features.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "names", tuple(self.names))
        if not self.values or len(self.values) != len(self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("feature values and unique names must align")
        for value in self.values:
            finite(value, "feature")


@dataclass(frozen=True)
class SourceMemory:
    memory_id: str
    source_task_id: str
    source_trace_id: str
    source_checkpoint_hash: str
    mechanism: str
    problem_action: dict[str, Any]
    correction: dict[str, Any]
    positive_conditions: tuple[dict[str, Any], ...] = ()
    exclusion_conditions: tuple[dict[str, Any], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "source-memory.v2"

    def __post_init__(self) -> None:
        if not all((self.source_task_id, self.source_trace_id, self.source_checkpoint_hash, self.mechanism)):
            raise ValueError("source memory requires provenance and mechanism")
        for key in ("problem_action", "correction", "evidence"):
            object.__setattr__(self, key, copy.deepcopy(getattr(self, key)))
        for key in ("positive_conditions", "exclusion_conditions"):
            object.__setattr__(self, key, tuple(copy.deepcopy(getattr(self, key))))
        canonical(self)

    def content_hash(self) -> str:
        value = plain(self)
        value.pop("memory_id")
        return digest(value)


@dataclass(frozen=True)
class StudySpec:
    study_id: str = "development"
    base_seed: int = 42
    replicates: int = 3
    max_steps: int = 8
    max_model_calls: int = 9
    budget_mode: Literal["continuation", "total"] = "continuation"
    lambda_inference: float = 0.01
    lambda_tool: float = 0.0
    inference_weights: dict[str, float] = field(default_factory=lambda: {"model_calls": 1.0})
    tool_weights: dict[str, float] = field(default_factory=lambda: {"tool_calls": 1.0})
    risk_budget: float = 0.05
    compute_budget: float = 2.0
    kl_weight: float = 0.01
    reminder_probability: float = 0.5
    schema_version: str = "study-spec.v2"

    def __post_init__(self) -> None:
        for key in ("base_seed", "replicates", "max_steps", "max_model_calls"):
            v = getattr(self, key)
            if type(v) is not int or v < (0 if key == "base_seed" else 1):
                raise ValueError(f"invalid {key}")
        if not self.study_id or self.budget_mode not in ("continuation", "total"):
            raise ValueError("invalid study ID/budget mode")
        for key in ("lambda_inference", "lambda_tool", "risk_budget", "compute_budget", "kl_weight"):
            finite(getattr(self, key), key, minimum=0)
        if self.risk_budget > 1 or not 0 < finite(self.reminder_probability, "reminder_probability") < 1:
            raise ValueError("risk/probability out of range")
        for key in ("inference_weights", "tool_weights"):
            weights = copy.deepcopy(getattr(self, key))
            if not weights:
                raise ValueError("cost weights must be explicit")
            for name, value in weights.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("invalid cost name")
                finite(value, name, minimum=0)
            object.__setattr__(self, key, weights)


def weighted_cost(costs: dict[str, float | None], weights: dict[str, float]) -> float | None:
    result = 0.0
    for name, weight in weights.items():
        if weight == 0:
            continue
        value = costs.get(name)
        if value is None:
            return None
        result += weight * finite(value, name, minimum=0)
    return result


def utility(success: bool, costs: dict[str, float | None], spec: StudySpec) -> float | None:
    if type(success) is not bool:
        raise ValueError("success must be bool")
    inference = weighted_cost(costs, spec.inference_weights) if spec.lambda_inference else 0.0
    tools = weighted_cost(costs, spec.tool_weights) if spec.lambda_tool else 0.0
    if inference is None or tools is None:
        return None
    return float(success) - spec.lambda_inference * inference - spec.lambda_tool * tools


def stage_seed(base_seed: int, trial_id: str, replicate: int, stage: str, step: int = 0) -> int:
    return int(digest([base_seed, trial_id, replicate, stage, step])[:15], 16) % (2**31 - 1)
