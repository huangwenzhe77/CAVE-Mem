"""Source repair evidence with explicit endpoints and independent confirmation.

Source replays are conditional on a checkpoint; the confidence statement does
not assert transfer to another task. Exceptions never become zero-reward tests.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, replace
from statistics import mean
from typing import Any, Callable

from .contracts import (PublicContext, SourceMemory, StudySpec, canonical, digest,
                        finite, plain, stage_seed, utility)


@dataclass(frozen=True)
class EvidenceSpec:
    endpoint: str = "terminal_utility"
    scope: str = "source_checkpoint_terminal"
    confidence: float = 0.95
    acceptance_epsilon: float = 0.0
    min_confirmation_pairs: int = 8
    max_steps: int = 12
    interval: str = "hoeffding"
    population_size: int | None = None
    require_confirmation: bool = True
    utility_protocol: dict = field(default_factory=lambda: plain(StudySpec()))
    cost_upper_bounds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.endpoint not in ("terminal_utility", "terminal_success", "local_test") or not self.scope:
            raise ValueError("declare a supported endpoint and evidence scope")
        if not 0 < finite(self.confidence, "confidence") < 1:
            raise ValueError("confidence must be inside (0,1)")
        finite(self.acceptance_epsilon, "acceptance_epsilon", minimum=0)
        if self.interval not in ("hoeffding", "exhaustive_finite_scope"):
            raise ValueError("unsupported interval method")
        for name in ("min_confirmation_pairs", "max_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if self.interval == "exhaustive_finite_scope" and (type(self.population_size) is not int or self.population_size < 1):
            raise ValueError("exhaustive scope requires its finite population size")
        object.__setattr__(self, "utility_protocol", plain(StudySpec(**self.utility_protocol)))
        object.__setattr__(self, "cost_upper_bounds", copy.deepcopy(self.cost_upper_bounds))
        for name, bound in self.cost_upper_bounds.items():
            finite(bound, name + " upper bound", minimum=0)
        _score_bounds(self)


def _score_bounds(spec: EvidenceSpec) -> tuple[float, float]:
    if spec.endpoint != "terminal_utility":
        return 0.0, 1.0
    protocol = StudySpec(**spec.utility_protocol)
    limits = {"model_calls": float(min(spec.max_steps - 1, protocol.max_model_calls)), "tool_calls": float(spec.max_steps),
              "gate_calls": 0.0, "retrieval_calls": 0.0, "feature_calls": 0.0, "encoder_calls": 0.0,
              **spec.cost_upper_bounds}
    penalty = 0.0
    for coefficient, weights in ((protocol.lambda_inference, protocol.inference_weights),
                                 (protocol.lambda_tool, protocol.tool_weights)):
        for name, weight in weights.items():
            if coefficient and weight:
                if name not in limits:
                    raise ValueError("utility evidence requires a declared finite upper bound for " + name)
                penalty += coefficient * weight * limits[name]
    return -penalty, 1.0


def locate_candidates(steps: list[dict], *, limit: int = 3) -> list[dict]:
    """Find falsifiable invalid/no-op/repeated-action candidates, not causes.

    Proposed corrections must come from observed admissible alternatives or an
    explicitly supplied repair; expert actions are source-only annotations.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    found, seen = [], set()
    for index, step in enumerate(steps):
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        key = canonical(action)
        reasons = []
        if step.get("valid") is False:
            reasons.append("invalid_action")
        if step.get("observation") == step.get("next_observation") and "next_observation" in step:
            reasons.append("no_observable_progress")
        if key in seen:
            reasons.append("repeated_action")
        seen.add(key)
        if not reasons:
            continue
        alternatives = [copy.deepcopy(a) for a in step.get("admissible_actions", [])
                        if isinstance(a, dict) and a != action]
        correction = step.get("correction")
        if correction is not None and not isinstance(correction, dict):
            raise ValueError("correction must be a typed action")
        found.append({"step_index": index, "problem_action": copy.deepcopy(action),
                      "correction": copy.deepcopy(correction), "alternatives": alternatives,
                      "reasons": reasons, "causal_status": "hypothesis_only"})
        if len(found) >= limit:
            break
    return found


def replay_source(env_factory: Callable, actor_factory: Callable, checkpoint: dict,
                  problem_action: dict, correction: dict, *, replicates: int = 40,
                  base_seed: int = 42, cohort: str = "confirmation",
                  spec: EvidenceSpec | None = None, local_test: Callable | None = None) -> list[dict]:
    """Restore both branches and execute a repair followed by a fixed actor.

    `local_test(public_context, public_result)` is a read-only executable [0,1]
    check. It never receives an environment or private checkpoint. The default
    endpoint is Eq. 6 terminal utility; success/local scores are distinct scopes.
    """
    spec = spec or EvidenceSpec()
    if type(replicates) is not int or replicates < 1 or not cohort:
        raise ValueError("declare positive replicates and a cohort")
    if spec.endpoint == "local_test" and local_test is None:
        raise ValueError("local_test endpoint needs an executable check")
    cp_hash = digest(checkpoint)
    utility_spec = StudySpec(**spec.utility_protocol)
    lower_score, upper_score = _score_bounds(spec)

    def empty_costs():
        return {"model_calls": 0.0, "physical_attempts": 0.0, "tool_calls": 0.0, "gate_calls": 0.0,
                "retrieval_calls": 0.0, "feature_calls": 0.0, "encoder_calls": 0.0,
                "input_tokens": 0.0, "output_tokens": 0.0, "api_cost": 0.0,
                "tool_cost": 0.0, "wall_seconds": 0.0}

    def add_usage(costs, usage):
        for key in ("model_calls", "physical_attempts", "input_tokens", "output_tokens", "api_cost"):
            measured = usage.get(key)
            costs[key] = None if measured is None or costs[key] is None else costs[key] + finite(measured, key, minimum=0)

    output = []
    for replicate in range(replicates):
        seed = stage_seed(base_seed, cp_hash, replicate, f"source:{cohort}")
        row = {"sample_id": digest([cp_hash, cohort, seed]), "checkpoint_hash": cp_hash,
               "replicate": replicate, "seed": seed, "cohort": cohort,
               "endpoint": spec.endpoint, "scope": spec.scope, "arms": {}, "valid": False,
               "source_protocol_hash": digest(spec), "problem_action": copy.deepcopy(problem_action),
               "correction": copy.deepcopy(correction), "problem_action_hash": digest(problem_action),
               "correction_hash": digest(correction), "score_bounds": [lower_score, upper_score]}
        sessions = []
        for name, action in (("original", problem_action), ("corrected", correction)):
            env = actor = None
            costs = empty_costs()
            started = time.perf_counter()
            arm = {"status": "error", "score": None, "success": None, "events": [],
                   "error": None, "costs": costs, "actor_calls": []}
            try:
                env, actor = env_factory(), actor_factory()
                if any(env is previous[0] or actor is previous[1] for previous in sessions):
                    raise ValueError("source factories must produce independent sessions")
                sessions.append((env, actor))
                if not env.capabilities().get("restore_certified"):
                    raise ValueError("source replay requires certified environment restoration")
                if not actor.capabilities().get("stateless"):
                    raise ValueError("source replay requires a frozen stateless actor")
                arm["capabilities"] = {"environment": plain(env.capabilities()), "actor": plain(actor.capabilities())}
                context = env.restore(copy.deepcopy(checkpoint))
                if not isinstance(context, PublicContext):
                    raise TypeError("source continuation requires a public context")
                if digest(env.snapshot()) != cp_hash:
                    raise ValueError("source checkpoint restoration mismatch")
                arm["restore_hash"] = env.state_fingerprint()
                arm["environment_config"] = env.configuration()
                arm["actor_config"] = actor.configuration()
                selected = copy.deepcopy(action)
                for step in range(spec.max_steps):
                    valid, reason = env.validate_action(copy.deepcopy(selected))
                    if not valid:
                        arm.update(status="invalid_action", score=0.0, success=False, error=reason)
                        break
                    costs["tool_calls"] += 1
                    result = env.step(copy.deepcopy(selected), seed=stage_seed(seed, cp_hash, replicate, "environment", step))
                    costs["tool_cost"] = None if costs["tool_cost"] is None or result.tool_cost is None else costs["tool_cost"] + finite(result.tool_cost, "tool_cost", minimum=0)
                    arm["events"].append({"action": selected, "result": plain(result)})
                    if not result.valid:
                        arm.update(status="invalid_action", score=0.0, success=False, error=result.error)
                        break
                    if spec.endpoint == "local_test":
                        before = env.state_fingerprint()
                        public_result = {"context": plain(result.context), "valid": result.valid,
                                         "terminal": result.terminal, "success": result.success,
                                         "action": copy.deepcopy(selected)}
                        score = finite(local_test(copy.deepcopy(result.context), public_result), "local_test score", minimum=0)
                        if score > 1 or before != env.state_fingerprint():
                            raise ValueError("local check must be read-only and bounded by 1")
                        arm.update(status="local_test_complete", score=score, success=result.success)
                        break
                    if result.terminal:
                        if type(result.success) is not bool:
                            raise ValueError("terminal success not supplied")
                        arm.update(status="terminal", score=float(result.success), success=result.success)
                        break
                    context = result.context
                    if context.remaining_steps <= 0 or costs["model_calls"] >= utility_spec.max_model_calls:
                        arm.update(status="budget_exhausted", score=0.0, success=False)
                        break
                    if step + 1 < spec.max_steps:
                        response = actor.propose(context, stage="source_continuation",
                                                 seed=stage_seed(seed, cp_hash, replicate, "actor", step),
                                                 reminder=None, call_id=f"source-{cohort}-{replicate}-{name}-{step}")
                        add_usage(costs, response.usage)
                        arm["actor_calls"].append(plain(response))
                        selected = copy.deepcopy(response.action)
                else:
                    arm.update(status="budget_exhausted", score=0.0, success=False)
                if arm["status"] == "error" and arm["error"] is None:
                    arm.update(status="budget_exhausted", score=0.0, success=False)
                if actor.configuration() != arm["actor_config"] or env.configuration() != arm["environment_config"]:
                    raise ValueError("source actor/environment configuration changed")
                costs["wall_seconds"] = time.perf_counter() - started
                if spec.endpoint == "terminal_utility":
                    arm["score"] = utility(arm["success"], costs, utility_spec)
                    if arm["score"] is None:
                        raise ValueError("source terminal utility has unknown required costs")
                for cost_name, bound in spec.cost_upper_bounds.items():
                    if costs.get(cost_name) is None or costs[cost_name] > bound:
                        raise ValueError("source registered cost bound unavailable/exceeded: " + cost_name)
                if not lower_score <= arm["score"] <= upper_score:
                    raise ValueError("source endpoint outside its registered bounds")
            except Exception as exc:
                arm.update(status="error", score=None, success=None, error=f"{type(exc).__name__}: {exc}")
                attempt = getattr(exc, "attempt", None)
                if attempt is not None:
                    arm["failed_attempt"] = plain(attempt)
                    add_usage(costs, attempt.get("usage", {}))
                elif costs["tool_calls"]:
                    costs["tool_cost"] = None
            finally:
                if arm["status"] == "error":
                    costs["wall_seconds"] = time.perf_counter() - started
                if env is not None:
                    verifier = getattr(env, "verifier_report", None)
                    if verifier is not None:
                        arm["private_verifier"] = {"report": copy.deepcopy(verifier), "sha256": digest(verifier)}
                    try:
                        env.close()
                    except Exception:
                        arm["cleanup_error"] = True
            row["arms"][name] = arm
        original, corrected = row["arms"]["original"], row["arms"]["corrected"]
        row["valid"] = (all(a["status"] != "error" for a in (original, corrected))
                        and original.get("restore_hash") == corrected.get("restore_hash")
                        and original.get("environment_config") == corrected.get("environment_config")
                        and original.get("actor_config") == corrected.get("actor_config")
                        and corrected["status"] != "invalid_action")
        row["record_hash"] = digest(row)
        output.append(row)
    return output


def evaluate_evidence(confirmation: list[dict], *, spec: EvidenceSpec | None = None,
                      screening_ids: tuple[str, ...] = (), candidate_count: int = 1,
                      claim: dict | None = None) -> dict:
    """Bound the conditional mean paired contrast on independent confirmation.

    Hoeffding uses the registered bounded contrast, not mean minus one SE.
    Exhaustive finite-scope evidence requires unique exhaustive units and
    supports only that declared finite scope; no population CI is invented.
    """
    spec = spec or EvidenceSpec()
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    ids = [r.get("sample_id") for r in confirmation]
    report = {"schema_version": "source-evidence.v2", "spec": plain(spec),
              "confirmation": copy.deepcopy(confirmation), "screening_ids": list(screening_ids),
              "candidate_count": candidate_count, "n": len(confirmation), "v": "?",
              "mean_difference": None, "lower_bound": None, "upper_bound": None,
              "reason": "insufficient_confirmation", "scope": spec.scope, "endpoint": spec.endpoint}
    if claim is not None:
        report["claim"] = copy.deepcopy(claim)

    def finish():
        report["evidence_hash"] = digest({k: v for k, v in report.items() if k != "evidence_hash"})
        return report

    if not confirmation:
        return finish()
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError("confirmation sample IDs must be unique and nonempty")
    if set(ids) & set(screening_ids):
        raise ValueError("screening and confirmation samples overlap")
    differences, configurations = [], set()
    lower_score, upper_score = _score_bounds(spec)
    utility_spec = StudySpec(**spec.utility_protocol)
    source_identity = None
    for row in confirmation:
        if row.get("record_hash") != digest({k: v for k, v in row.items() if k != "record_hash"}):
            raise ValueError("source replay checksum mismatch")
        if row.get("endpoint") != spec.endpoint or row.get("scope") != spec.scope:
            raise ValueError("evidence endpoint/scope mismatch")
        if spec.require_confirmation and row.get("cohort") != "confirmation":
            raise ValueError("confirmation cohort is required")
        if row.get("source_protocol_hash") != digest(spec) or row.get("score_bounds") != [lower_score, upper_score]:
            raise ValueError("source replay protocol/bounds mismatch")
        if row.get("sample_id") != digest([row.get("checkpoint_hash"), row.get("cohort"), row.get("seed")]):
            raise ValueError("source sample identity mismatch")
        if row.get("problem_action_hash") != digest(row["problem_action"]) or row.get("correction_hash") != digest(row["correction"]):
            raise ValueError("source replay action checksum mismatch")
        identity = (row["checkpoint_hash"], row["problem_action_hash"], row["correction_hash"])
        if source_identity is not None and source_identity != identity:
            raise ValueError("confirmation mixes source checkpoints or tested actions")
        source_identity = identity
        if row.get("valid") is not True:
            report["reason"] = "invalid_or_failed_source_replay"
            return finish()
        scores = []
        for name in ("original", "corrected"):
            arm = row["arms"][name]
            if arm["status"] == "error" or not arm["capabilities"]["environment"].get("restore_certified") or not arm["capabilities"]["actor"].get("stateless"):
                raise ValueError("confirmation lacks certified restoration/frozen actor")
            configurations.add(digest([arm["environment_config"], arm["actor_config"]]))
            score = finite(arm["score"], "score")
            if not lower_score <= score <= upper_score:
                raise ValueError("source endpoint outside registered score bounds")
            if spec.endpoint == "terminal_utility":
                recomputed = utility(arm["success"], arm["costs"], utility_spec)
                if recomputed is None or not math.isclose(recomputed, score, rel_tol=0, abs_tol=1e-10):
                    raise ValueError("source utility/cost mismatch")
            elif spec.endpoint == "terminal_success" and score != float(arm["success"]):
                raise ValueError("source success endpoint mismatch")
            scores.append(score)
        differences.append(scores[1] - scores[0])
    if len(configurations) != 1:
        raise ValueError("source confirmation changed actor/environment configuration")
    avg = mean(differences)
    if spec.interval == "exhaustive_finite_scope":
        unit_ids = [r.get("exhaustive_unit_id") for r in confirmation]
        if len(confirmation) != spec.population_size or None in unit_ids or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("exhaustive finite scope needs every uniquely identified unit")
        lower = upper = avg
    else:
        # A paired contrast has range [-width,width]; use a two-sided bound.
        width = upper_score - lower_score
        radius = width * math.sqrt(2.0 * math.log(2.0 / (1.0 - spec.confidence)) / len(differences))
        lower, upper = max(-width, avg - radius), min(width, avg + radius)
    report.update(mean_difference=avg, lower_bound=lower, upper_bound=upper)
    if len(differences) < spec.min_confirmation_pairs:
        return finish()
    passed = lower > spec.acceptance_epsilon
    report.update(v=1 if passed else 0,
                  reason="supported_in_declared_source_scope" if passed else "improvement_not_supported")
    return finish()


def validate_evidence(memory: SourceMemory) -> dict:
    evidence = memory.evidence
    if evidence.get("schema_version") != "source-evidence.v2":
        return {"v": "?", "reason": "unsupported_or_missing_evidence", "scope": None}
    spec_fields = evidence.get("spec", {})
    try:
        spec = EvidenceSpec(**spec_fields)
        checked = evaluate_evidence(evidence.get("confirmation", []), spec=spec,
                                    screening_ids=tuple(evidence.get("screening_ids", [])),
                                    candidate_count=evidence.get("candidate_count", 1),
                                    claim=evidence.get("claim"))
        if evidence.get("claim") != _memory_claim(memory):
            raise ValueError("memory mechanism/conditions differ from the registered evidence claim")
        if any(r.get("checkpoint_hash") != memory.source_checkpoint_hash for r in checked["confirmation"]):
            raise ValueError("memory and source checkpoint mismatch")
        for row in checked["confirmation"]:
            if row["problem_action_hash"] != digest(memory.problem_action) or row["correction_hash"] != digest(memory.correction):
                raise ValueError("memory actions differ from the source-tested intervention")
        if canonical(evidence) != canonical(checked):
            raise ValueError("evidence statistics or checksum mismatch")
    except (ValueError, TypeError, KeyError) as exc:
        return {"v": 0, "reason": f"invalid_evidence:{exc}", "scope": evidence.get("scope")}
    return {"v": checked["v"], "reason": checked["reason"], "scope": checked["scope"]}


def validity(memory: SourceMemory | None):
    return "?" if memory is None else validate_evidence(memory)["v"]


def _memory_claim(memory: SourceMemory) -> dict:
    return {"source_task_id": memory.source_task_id, "source_trace_id": memory.source_trace_id,
            "checkpoint_hash": memory.source_checkpoint_hash, "problem_action": memory.problem_action,
            "correction": memory.correction, "mechanism_hypothesis": memory.mechanism,
            "positive_conditions": plain(memory.positive_conditions),
            "exclusion_conditions": plain(memory.exclusion_conditions),
            "claim_boundary": "repair_support_in_declared_scope;mechanism_and_transfer_remain_hypotheses"}


def build_memory(*, source_task_id: str, source_trace_id: str, source_checkpoint_hash: str,
                 mechanism: str, problem_action: dict, correction: dict, evidence: dict,
                 positive_conditions: tuple[dict, ...] = (), exclusion_conditions: tuple[dict, ...] = ()) -> SourceMemory:
    memory = SourceMemory("pending", source_task_id, source_trace_id, source_checkpoint_hash,
                          mechanism, problem_action, correction, positive_conditions, exclusion_conditions, evidence)
    if evidence.get("schema_version") == "source-evidence.v2":
        if evidence.get("claim") is not None and evidence["claim"] != _memory_claim(memory):
            raise ValueError("cannot rebind existing source evidence to a different registered claim")
        bound_evidence = evaluate_evidence(evidence.get("confirmation", []),
                                          spec=EvidenceSpec(**evidence["spec"]),
                                          screening_ids=tuple(evidence.get("screening_ids", [])),
                                          candidate_count=evidence.get("candidate_count", 1),
                                          claim=_memory_claim(memory))
        # Binding a claim must not silently accept an edited/invalid report.
        original = evaluate_evidence(evidence.get("confirmation", []),
                                    spec=EvidenceSpec(**evidence["spec"]),
                                    screening_ids=tuple(evidence.get("screening_ids", [])),
                                    candidate_count=evidence.get("candidate_count", 1),
                                    claim=evidence.get("claim"))
        if canonical(original) != canonical(evidence):
            raise ValueError("cannot bind a memory to a tampered evidence report")
        memory = replace(memory, evidence=bound_evidence)
    return replace(memory, memory_id="mem-" + memory.content_hash())
