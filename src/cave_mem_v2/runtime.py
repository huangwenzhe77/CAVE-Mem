"""Baseline-first single-intervention collectors with explicit budget semantics.

The cached draft is produced once, before assignment. Both paired continuations
receive it as data, rather than relying on repeat provider samples matching.
The collector never passes private snapshots to actors or feature builders.
"""
from __future__ import annotations

import copy
import inspect
import math
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .contracts import (FeatureRecord, PublicContext, SourceMemory, StudySpec,
                        digest, finite, plain, read_json, stage_seed, utility, weighted_cost, write_json)
from .environments import Environment, EnvStep
from .providers import Actor, ActorResponse, ProviderError


class ProtocolError(ValueError):
    pass


def callable_identity(callback) -> dict:
    code = getattr(callback, "__code__", None)
    return {"name": getattr(callback, "__qualname__", type(callback).__qualname__),
            "module": getattr(callback, "__module__", type(callback).__module__),
            "version": getattr(callback, "version", getattr(getattr(callback, "__self__", None), "version", None)),
            "bytecode": digest(code.co_code.hex()) if code is not None else None}


def bounded_context(context: PublicContext, steps: int) -> PublicContext:
    return replace(context, remaining_steps=min(context.remaining_steps, max(0, steps)))


def empty_costs() -> dict[str, float | None]:
    return {"model_calls": 0.0, "physical_attempts": 0.0, "tool_calls": 0.0, "gate_calls": 0.0,
            "retrieval_calls": 0.0, "feature_calls": 0.0, "encoder_calls": 0.0,
            "input_tokens": 0.0, "output_tokens": 0.0, "api_cost": 0.0,
            "tool_cost": 0.0, "wall_seconds": 0.0}


def add_costs(left: dict, right: dict) -> dict:
    """Unknown measured costs remain unknown, including failed attempts."""
    result = copy.deepcopy(left)
    for key in set(left) | set(right):
        # An omitted field is not a reported zero. Common fields missing from
        # either actual measurement become unknown; stage-only counters default
        # to zero through the caller's explicitly initialized ledger.
        a, b = left.get(key, 0.0), right.get(key, 0.0)
        result[key] = None if a is None or b is None else finite(a, key, minimum=0) + finite(b, key, minimum=0)
    return result


def response_costs(response: ActorResponse) -> dict:
    costs = empty_costs()
    for key in ("model_calls", "physical_attempts", "input_tokens", "output_tokens", "api_cost"):
        costs[key] = response.usage.get(key)
    costs["wall_seconds"] = sum(float(a.get("wall_seconds", 0.0)) for a in response.attempts)
    return costs


@dataclass(frozen=True)
class PreparedDecision:
    trial: dict
    checkpoint: dict = field(repr=False)
    draft: ActorResponse = field(repr=False)
    common_costs: dict = field(repr=False)
    preparation_events: tuple[dict, ...] = field(repr=False)
    state_fingerprint: str = ""
    environment_config: dict = field(default_factory=dict)
    actor_config: dict = field(default_factory=dict)
    actor_capabilities: dict = field(default_factory=dict)
    environment_capabilities: dict = field(default_factory=dict)


def default_checks(context: PublicContext, action: dict, memory: SourceMemory) -> dict:
    """Default integration defers to source validators; no duplicated eligibility rule."""
    from .features import assess
    return assess(context, action, memory)


def default_features(context: PublicContext, action: dict, memory: SourceMemory, checks: dict,
                     spec: StudySpec, *, retrieval_score: float = 0.0):
    from .features import FeatureBuilder
    builder = FeatureBuilder()
    record = builder.build(context, action, memory, checks,
                           remaining_calls=spec.max_model_calls, retrieval_score=retrieval_score)
    return record, builder.last_usage


class InterventionCollector:
    def __init__(self, spec: StudySpec | None = None, *, artifact_dir: str | Path | None = None,
                 check_builder: Callable | None = None, feature_builder: Callable | None = None) -> None:
        self.spec = spec or StudySpec()
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.check_builder = check_builder or default_checks
        self.feature_builder = feature_builder or default_features
        self._prepared: dict[str, PreparedDecision] = {}
        self._prepared_hashes: dict[str, str] = {}
        self._new_preparations: set[str] = set()

    def _seed(self, trial_id: str, replicate: int, stage: str, step: int = 0) -> int:
        return stage_seed(self.spec.base_seed, trial_id, replicate, stage, step)

    def prepare(self, env: Environment, actor: Actor, memory: SourceMemory | Callable | None, *, trial_id: str,
                replicate: int = 0, group_keys: tuple[str, ...] = (), sampling_weight: float = 1.0,
                common_overhead: dict | None = None) -> PreparedDecision:
        if type(replicate) is not int or not 0 <= replicate < self.spec.replicates:
            raise ValueError("replicate out of range")
        if not trial_id or finite(sampling_weight, "sampling_weight", minimum=0) == 0:
            raise ValueError("trial id and positive sampling weight are required")
        if hasattr(env, "terminal_status") and env.terminal_status()[0]:
            raise ProtocolError("target checkpoint is already terminal")
        context = bounded_context(env.observe(), self.spec.max_steps)
        if context.remaining_steps == 0:
            raise ProtocolError("target checkpoint has no execution budget")
        checkpoint = copy.deepcopy(env.snapshot())
        checkpoint_hash = digest(checkpoint)
        before = env.state_fingerprint()
        actor_config = copy.deepcopy(actor.configuration())
        env_config = copy.deepcopy(env.configuration())
        request_signature = digest({"checkpoint": checkpoint_hash, "actor": actor_config,
                                    "environment": env_config, "protocol": plain(self.spec),
                                    "memory": memory.content_hash() if isinstance(memory, SourceMemory) else None,
                                    "selector": getattr(memory, "version", None) if callable(memory) else None,
                                    "unversioned_selector": callable(memory) and not getattr(memory, "version", None),
                                    "feature_builder": callable_identity(self.feature_builder),
                                    "check_builder": callable_identity(self.check_builder),
                                    "groups": sorted(group_keys),
                                    "sampling_weight": sampling_weight, "common_overhead": common_overhead})
        if not actor.capabilities().get("stateless"):
            raise ProtocolError("collector requires a stateless actor with explicit public history")
        if trial_id in self._prepared:
            cached = self._prepared[trial_id]
            if (cached.state_fingerprint != before or cached.actor_config != actor_config
                    or cached.environment_config != env_config
                    or cached.trial["request_signature"] != request_signature):
                raise ProtocolError("trial ID reused with changed checkpoint or actor/environment configuration")
            self._verify_prepared(cached)
            return cached
        prep_path = self.artifact_dir / "private" / (digest(trial_id)[:24] + ".preparation.json") if self.artifact_dir else None
        if prep_path and prep_path.exists():
            saved = read_json(prep_path)
            if callable(memory) and not getattr(memory, "version", None):
                raise ProtocolError("cross-process selector resume requires the caller to freeze the candidate explicitly")
            if saved.get("preparation_hash") != digest(saved.get("prepared")):
                raise ProtocolError("cached preparation checksum mismatch")
            body = saved["prepared"]
            if body["trial"].get("request_signature") != request_signature:
                raise ProtocolError("cached preparation inputs changed; use a new trial/run")
            body["draft"] = ActorResponse(**body["draft"])
            body["preparation_events"] = tuple(body["preparation_events"])
            restored = PreparedDecision(**body)
            self._prepared[trial_id] = restored
            self._prepared_hashes[trial_id] = digest(restored)
            return restored
        # Only the ordinary public context enters the untreated draft.
        draft_started = time.perf_counter()
        draft = actor.propose(copy.deepcopy(context), stage="draft", seed=self._seed(trial_id, 0, "draft"),
                              reminder=None, call_id=f"{trial_id}:draft")
        if env.state_fingerprint() != before:
            raise ProtocolError("draft generation changed environment state")
        if actor.configuration() != actor_config:
            raise ProtocolError("actor configuration changed during draft")
        costs = response_costs(draft)
        costs["wall_seconds"] = time.perf_counter() - draft_started
        overhead = empty_costs()
        # CPU retrieval/check/features overhead is separately timed and real API
        # encoder usage may be supplied explicitly by a metered caller.
        started = time.perf_counter()
        retrieval_score = 0.0
        shared_usage = empty_costs()
        if callable(memory):
            selector = memory
            selected = selector(copy.deepcopy(context))
            if isinstance(selected, tuple):
                if len(selected) == 3:
                    memory, retrieval_score, usage = selected
                    shared_usage = add_costs(shared_usage, usage)
                else:
                    memory, retrieval_score = selected
                    if hasattr(selector, "last_usage"):
                        shared_usage = add_costs(shared_usage, selector.last_usage)
                    elif getattr(selector, "encoder", None) is not None:
                        shared_usage["api_cost"] = None
            else:
                memory = selected
            if memory is not None and not isinstance(memory, SourceMemory):
                raise TypeError("memory selector must return SourceMemory, (memory, score), or None")
        if memory is None:
            checks = {"v": 0, "rho": "?", "e": "?", "reason": "no_candidate"}
            features = FeatureRecord((0.0,), ("no_candidate",))
        else:
            checks = self.check_builder(copy.deepcopy(context), copy.deepcopy(draft.action), copy.deepcopy(memory))
            parameters = inspect.signature(self.feature_builder).parameters
            kwargs = {"retrieval_score": retrieval_score} if "retrieval_score" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()) else {}
            features = self.feature_builder(copy.deepcopy(context), copy.deepcopy(draft.action), copy.deepcopy(memory), copy.deepcopy(checks), self.spec, **kwargs)
            if isinstance(features, tuple):
                features, usage = features
                shared_usage = add_costs(shared_usage, usage)
            elif hasattr(self.feature_builder, "last_usage"):
                shared_usage = add_costs(shared_usage, self.feature_builder.last_usage)
            elif self.feature_builder is not default_features:
                # An arbitrary callback may use a paid encoder. Local callers
                # can return explicit zero/API usage rather than assume it.
                shared_usage["api_cost"] = None
        if not isinstance(features, FeatureRecord):
            raise TypeError("feature builder must return FeatureRecord")
        overhead["wall_seconds"] = time.perf_counter() - started
        # Shared runtime wall timing already encloses callback CPU/encoder time.
        shared_usage.pop("wall_seconds", None)
        overhead = add_costs(overhead, shared_usage)
        if common_overhead is not None:
            overhead = add_costs(overhead, common_overhead)
        costs = add_costs(costs, overhead)
        if env.state_fingerprint() != before:
            raise ProtocolError("assessment changed environment state")
        keys = {"task:" + context.task_id, "checkpoint:" + checkpoint_hash, *group_keys}
        if memory:
            keys.update(("source_task:" + memory.source_task_id, "task:" + memory.source_task_id,
                         "source_trace:" + memory.source_trace_id, "memory:" + memory.content_hash()))
        trial = {"trial_id": trial_id, "task_id": context.task_id,
                 "checkpoint_id": checkpoint_hash, "context": plain(context),
                 "draft": {"action": copy.deepcopy(draft.action), "text": draft.text,
                           "prompt_hash": draft.prompt_hash, "action_hash": digest(draft.action)},
                 "memory": plain(memory) if memory else None, "checks": plain(checks),
                 "features": plain(features), "group_keys": sorted(keys),
                 "sampling_weight": sampling_weight, "retrieval_score": retrieval_score,
                 "request_signature": request_signature}
        events = ({"event": "draft_cached", "response": plain(draft)},
                  {"event": "candidate_selected", "memory_id": memory.memory_id if memory else None,
                   "retrieval_score": retrieval_score},
                  {"event": "features_frozen", "feature_hash": digest(features),
                   "context_hash": digest(context), "draft_action_hash": digest(draft.action)})
        prepared = PreparedDecision(trial, checkpoint, draft, costs, events, before, env_config,
                                    actor_config, copy.deepcopy(actor.capabilities()), copy.deepcopy(env.capabilities()))
        self._prepared[trial_id] = prepared
        self._prepared_hashes[trial_id] = digest(prepared)
        self._new_preparations.add(trial_id)
        if prep_path:
            write_json(prep_path, {"prepared": plain(prepared), "preparation_hash": digest(prepared)})
        return prepared

    def _verify_prepared(self, prepared: PreparedDecision) -> None:
        expected = self._prepared_hashes.get(prepared.trial["trial_id"])
        if expected is not None and expected != digest(prepared):
            raise ProtocolError("frozen prepared decision was mutated")

    def _restore(self, env: Environment, prepared: PreparedDecision) -> None:
        env.restore(copy.deepcopy(prepared.checkpoint))
        if env.configuration() != prepared.environment_config:
            raise ProtocolError("restored environment configuration changed")
        if env.state_fingerprint() != prepared.state_fingerprint:
            raise ProtocolError("environment restore fingerprint mismatch")
        if digest(bounded_context(env.observe(), self.spec.max_steps)) != digest(prepared.trial["context"]):
            raise ProtocolError("restored public context mismatch")

    @staticmethod
    def reminder(memory: dict, action: dict) -> str:
        evidence = memory.get("evidence", {})
        support = {key: evidence.get(key) for key in ("scope", "endpoint", "mean_difference", "lower_bound")}
        return (f"A source-tested failure mechanism was: {memory['mechanism']} "
                f"Before executing {plain(action)}, check whether its observable preconditions hold. "
                f"The source-side alternative was {memory['correction']}; adapt it to this task. "
                f"Source memory: {memory.get('memory_id')}; replay support: {support}.")

    @staticmethod
    def _check_response(response: ActorResponse, prepared: PreparedDecision) -> None:
        if response.model != prepared.draft.model:
            raise ProtocolError("provider-reported actor model changed after draft")
        if (response.fingerprint is not None and prepared.draft.fingerprint is not None
                and response.fingerprint != prepared.draft.fingerprint):
            raise ProtocolError("provider-reported model fingerprint changed after draft")

    def _arm(self, env: Environment, actor: Actor, prepared: PreparedDecision, z: int,
             replicate: int) -> dict:
        costs = copy.deepcopy(prepared.common_costs)
        incremental = empty_costs()
        events, actions = [], []
        status, success, error = "error", None, None
        trial_id = prepared.trial["trial_id"]
        started = time.perf_counter()
        try:
            if actor.configuration() != prepared.actor_config:
                raise ProtocolError("actor configuration differs across arms")
            context = bounded_context(env.observe(), self.spec.max_steps)
            proposal = prepared.draft
            total_calls = int(costs.get("model_calls") or 0)
            continuation_calls = 0
            if z:
                if prepared.trial["memory"] is None:
                    raise ProtocolError("cannot remind without a source memory")
                if self.spec.budget_mode == "total" and total_calls >= self.spec.max_model_calls:
                    status, success = "budget_exhausted", False
                    return {"success": success, "status": status, "costs": costs, "actions": actions,
                            "events": events, "error": error, "incremental_costs": incremental,
                            "utility": utility(success, costs, self.spec)}
                message = self.reminder(prepared.trial["memory"], proposal.action)
                proposal = actor.propose(copy.deepcopy(context), stage="revision", seed=self._seed(trial_id, replicate, "revision"),
                                         reminder=message, call_id=f"{trial_id}:{replicate}:revision")
                incremental = add_costs(incremental, response_costs(proposal))
                total_calls += 1
                events.append({"event": "proposal", "stage": "revision", "reminder": message, "response": plain(proposal)})
                self._check_response(proposal, prepared)
            # The first decision executes a cached/revised proposal, followed by
            # identical post-decision continuation allowance. Revision does not
            # consume that allowance in primary budget_mode=continuation.
            for step in range(self.spec.max_steps):
                valid, reason = env.validate_action(copy.deepcopy(proposal.action))
                if not valid:
                    status, success = "invalid_action", False
                    events.append({"event": "invalid_action", "action": plain(proposal.action), "reason": reason})
                    break
                incremental["tool_calls"] += 1
                result = env.step(copy.deepcopy(proposal.action), self._seed(trial_id, replicate, "environment", step))
                incremental["tool_cost"] = None if result.tool_cost is None or incremental["tool_cost"] is None else incremental["tool_cost"] + result.tool_cost
                actions.append(copy.deepcopy(proposal.action))
                events.append({"event": "execution", "step": step, "action": plain(proposal.action), "result": plain(result)})
                if not result.valid:
                    status, success = "invalid_action", False
                    break
                if result.terminal:
                    if type(result.success) is not bool:
                        raise ProtocolError("terminal outcome lacks binary success")
                    status, success = "terminal", result.success
                    break
                if step + 1 >= self.spec.max_steps or result.context.remaining_steps <= 0:
                    status, success = "budget_exhausted", False
                    break
                if ((self.spec.budget_mode == "total" and total_calls >= self.spec.max_model_calls)
                        or (self.spec.budget_mode == "continuation" and continuation_calls >= self.spec.max_model_calls)):
                    status, success = "budget_exhausted", False
                    break
                context = bounded_context(result.context, self.spec.max_steps - step - 1)
                proposal = actor.propose(copy.deepcopy(context), stage="continuation",
                                         seed=self._seed(trial_id, replicate, "continuation", step + 1),
                                         reminder=None, call_id=f"{trial_id}:{replicate}:continuation:{step + 1}")
                total_calls += 1
                continuation_calls += 1
                incremental = add_costs(incremental, response_costs(proposal))
                events.append({"event": "proposal", "stage": "continuation", "reminder": None, "response": plain(proposal)})
                self._check_response(proposal, prepared)
            if actor.configuration() != prepared.actor_config:
                raise ProtocolError("actor configuration changed during continuation")
        except Exception as exc:
            status, success, error = "error", None, {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, ProviderError):
                events.append({"event": "provider_error", "attempt": exc.attempt})
                failed_costs = empty_costs()
                for key in ("model_calls", "physical_attempts", "input_tokens", "output_tokens", "api_cost"):
                    failed_costs[key] = exc.attempt.get("usage", {}).get(key)
                incremental = add_costs(incremental, failed_costs)
            else:
                # A failed tool could have run before communication failed.
                incremental["tool_cost"] = None
        incremental["wall_seconds"] = time.perf_counter() - started
        costs = add_costs(costs, incremental)
        result = {"success": success, "status": status, "costs": costs,
                  "incremental_costs": incremental, "actions": actions, "events": events, "error": error}
        verifier = getattr(env, "verifier_report", None)
        if verifier is not None:
            result["private_verifier"] = {"report": copy.deepcopy(verifier), "sha256": digest(verifier)}
        result["utility"] = utility(success, costs, self.spec) if type(success) is bool else None
        return result

    def _assignment(self, prepared: PreparedDecision, replicate: int) -> dict:
        p = self.spec.reminder_probability
        assigned = int(random.Random(self._seed(prepared.trial["trial_id"], replicate, "assignment")).random() < p)
        return {"z": assigned, "propensity": p if assigned else 1 - p, "reminder_probability": p,
                "assignment_seed": self._seed(prepared.trial["trial_id"], replicate, "assignment")}

    def _base_record(self, prepared: PreparedDecision, replicate: int, *, logging_gate: bool = True) -> dict:
        record = {"trial": copy.deepcopy(prepared.trial), "protocol": plain(self.spec), "replicate": replicate,
                "common_costs": copy.deepcopy(prepared.common_costs), "preparation_events": plain(prepared.preparation_events),
                "actor_config": prepared.actor_config, "environment_config": prepared.environment_config,
                "capabilities": {"actor": prepared.actor_capabilities, "environment": prepared.environment_capabilities},
                "feature_hash": digest(prepared.trial["features"])}
        if logging_gate:
            started = time.perf_counter()
            record["assignment"] = self._assignment(prepared, replicate)
            costs = empty_costs()
            eligible = prepared.trial.get("memory") is not None and prepared.trial.get("checks", {}).get("v") == 1
            costs["gate_calls"] = float(eligible)
            costs["wall_seconds"] = time.perf_counter() - started
            record["logging_gate_costs"] = costs
            record["gate_implementation"] = "randomized_assignment_not_learned_policy"
            record["common_costs"] = add_costs(record["common_costs"], costs)
        return record

    def _save(self, record: dict) -> dict:
        record = plain(record)
        record["record_hash"] = digest(record)
        if self.artifact_dir is not None:
            path = self.artifact_dir / (record["record_id"] + ".json")
            if path.exists():
                raise FileExistsError("refusing to overwrite an experimental record; choose a new run directory")
            write_json(path, record)
        return record

    def _cached_record(self, record_id: str, input_hash: str) -> dict | None:
        if self.artifact_dir is None:
            return None
        path = self.artifact_dir / (record_id + ".json")
        if not path.exists():
            return None
        record = read_json(path)
        if record.get("record_hash") != digest({k: v for k, v in record.items() if k != "record_hash"}):
            raise ProtocolError("cached collection checksum mismatch")
        if record.get("input_hash") != input_hash:
            raise ProtocolError("cached collection inputs changed")
        return record

    def collect_prepared_pair(self, prepared: PreparedDecision, env_factory: Callable[[], Environment],
                              actor_factory: Callable[[], Actor], *, replicate: int = 0) -> dict:
        if type(replicate) is not int or not 0 <= replicate < self.spec.replicates:
            raise ValueError("replicate out of range")
        self._verify_prepared(prepared)
        record_id = digest([prepared.trial["trial_id"], replicate, "paired"])[:24]
        input_hash = digest([prepared, plain(self.spec), replicate])
        cached = self._cached_record(record_id, input_hash)
        if cached is not None:
            return cached
        envs = [env_factory(), env_factory()]
        actors = [actor_factory(), actor_factory()]
        if envs[0] is envs[1] or actors[0] is actors[1]:
            raise ProtocolError("paired factories must return independent sessions")
        record = self._base_record(prepared, replicate)
        record.update(schema_version="paired-intervention.v2", record_id=record_id, input_hash=input_hash,
                      coupling={"environment": prepared.environment_capabilities.get("future_randomness"),
                                "actor": prepared.actor_capabilities.get("sampling_coupling", "unspecified"),
                                "seed_scheme": "stage-step-keyed.v2"}, arms={}, valid=False, error=None)
        try:
            for env in envs:
                if not env.capabilities().get("restore_certified"):
                    raise ProtocolError("paired collection requires explicitly certified restoration")
            order = [0, 1]
            random.Random(self._seed(prepared.trial["trial_id"], replicate, "arm_order")).shuffle(order)
            eligible = prepared.trial["memory"] is not None and prepared.trial["checks"].get("v") == 1
            if not eligible:
                order = [0]
                record["schema_version"] = "ineligible-decision.v2"
            record["execution_order"] = order
            for z in order:
                self._restore(envs[z], prepared)
                metered = replace(prepared, common_costs=record["common_costs"])
                record["arms"]["remind" if z else "silent"] = self._arm(envs[z], actors[z], metered, z, replicate)
            self._verify_prepared(prepared)
            record["valid"] = eligible and all(arm["status"] != "error" for arm in record["arms"].values())
            record["exclusion_reasons"] = (["no_eligible_source_memory"] if not eligible else
                                           [] if record["valid"] else ["arm_execution_error"])
            if record["valid"]:
                silent, remind = record["arms"]["silent"], record["arms"]["remind"]
                c0, c1 = (weighted_cost(arm["costs"], self.spec.inference_weights) for arm in (silent, remind))
                record["labels"] = {"advantage": None if silent["utility"] is None or remind["utility"] is None else remind["utility"] - silent["utility"],
                                    "regression": int(silent["success"] and not remind["success"]),
                                    "rescue": int(not silent["success"] and remind["success"]),
                                    "incremental_cost": None if c0 is None or c1 is None else c1 - c0,
                                    "delta_success": int(remind["success"]) - int(silent["success"])}
        except Exception as exc:
            record.update(valid=False, exclusion_reasons=["preparation_or_restore_error"],
                          error={"type": type(exc).__name__, "message": str(exc)})
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception as exc:
                    record.setdefault("cleanup_errors", []).append(type(exc).__name__)
        is_new = prepared.trial["trial_id"] in self._new_preparations
        collection_costs = copy.deepcopy(prepared.common_costs) if is_new else empty_costs()
        collection_costs = add_costs(collection_costs, record.get("logging_gate_costs", {}))
        record["preparation_charged_in_collection"] = is_new
        self._new_preparations.discard(prepared.trial["trial_id"])
        for arm in record["arms"].values():
            collection_costs = add_costs(collection_costs, arm["incremental_costs"])
        record["collection_costs"] = collection_costs
        return self._save(record)

    def collect_case(self, env_factory: Callable[[], Environment], actor_factory: Callable[[], Actor],
                     memory: SourceMemory | Callable | None, *, trial_id: str, **kwargs) -> list[dict]:
        return [self.collect_pair(env_factory, actor_factory, memory, trial_id=trial_id,
                                  replicate=replicate, **kwargs) for replicate in range(self.spec.replicates)]

    def collect_pair(self, env_factory: Callable[[], Environment], actor_factory: Callable[[], Actor],
                     memory: SourceMemory | Callable | None, *, trial_id: str, replicate: int = 0,
                     group_keys: tuple[str, ...] = (), sampling_weight: float = 1.0,
                     checkpoint: dict | None = None, common_overhead: dict | None = None) -> dict:
        envs = [env_factory(), env_factory()]
        actors = [actor_factory(), actor_factory()]
        if envs[0] is envs[1] or actors[0] is actors[1]:
            raise ProtocolError("paired factories must return independent sessions")
        prepared = None
        try:
            for env in envs:
                if not env.capabilities().get("restore_certified"):
                    raise ProtocolError("paired collection requires explicitly certified environment restoration")
            if checkpoint is None:
                envs[0].reset(self._seed(trial_id, 0, "reset"))
            else:
                envs[0].restore(copy.deepcopy(checkpoint))
            prepared = self.prepare(envs[0], actors[0], memory, trial_id=trial_id, replicate=replicate,
                                    group_keys=group_keys, sampling_weight=sampling_weight, common_overhead=common_overhead)
            return self.collect_prepared_pair(prepared, env_factory, actor_factory, replicate=replicate)
        except Exception as exc:
            record = self._base_record(prepared, replicate) if prepared else {"trial": {"trial_id": trial_id}, "protocol": plain(self.spec), "replicate": replicate}
            record.update(schema_version="paired-intervention.v2", record_id=digest([trial_id, replicate, "paired"])[:24],
                          valid=False, arms={}, exclusion_reasons=["preparation_or_restore_error"],
                          error={"type": type(exc).__name__, "message": str(exc)})
            if isinstance(exc, ProviderError):
                record["failed_attempt"] = exc.attempt
            return self._save(record)
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass

    def collect_deployment(self, env: Environment, actor: Actor, memory: SourceMemory | Callable | None,
                           policy, *, trial_id: str, replicate: int = 0,
                           group_keys: tuple[str, ...] = (), sampling_weight: float = 1.0,
                           reset: bool = False, policy_id: str | None = None,
                           common_overhead: dict | None = None) -> dict:
        """Execute one frozen policy decision, never a mixture of paired outcomes.

        Its schema is intentionally rejected by randomized-training exporters.
        A deterministic gate or a policy probability is evaluated on public,
        frozen pre-treatment features, then exactly one continuation is run.
        """
        prepared = None
        try:
            if reset:
                env.reset(self._seed(trial_id, 0, "reset"))
            prepared = self.prepare(env, actor, memory, trial_id=trial_id, replicate=replicate,
                                    group_keys=group_keys, sampling_weight=sampling_weight, common_overhead=common_overhead)
            record = self._base_record(prepared, replicate, logging_gate=False)
            record.pop("assignment", None)
            record.update(schema_version="deployment-intervention.v2",
                          record_id=digest([trial_id, replicate, "deployment", policy_id or type(policy).__name__])[:24],
                          policy_name=policy_id or getattr(policy, "name", type(policy).__name__), valid=False, arms={}, error=None)
            if policy_id:
                policy_identity = policy_id
            elif getattr(policy, "network", None) is not None:
                policy_identity = digest({"network": policy.network.state(), "schema": policy.schema,
                                          "training_hash": getattr(policy, "training_hash", None),
                                          "safety": getattr(policy, "safety", {}),
                                          "force_silent": getattr(policy, "force_silent", False)})
            elif getattr(policy, "outcome_model", None) is not None:
                model = policy.outcome_model
                policy_identity = digest({"class": type(policy).__qualname__,
                                          "outcome_networks": [network.state() for network in model.networks],
                                          "schema": model.schema, "calibration": model.calibration,
                                          "risk_threshold": getattr(policy, "risk_threshold", None),
                                          "use_lower": getattr(policy, "use_lower", None)})
            else:
                policy_identity = digest({"class": type(policy).__qualname__, "name": getattr(policy, "name", None),
                                          "configuration": policy.configuration() if hasattr(policy, "configuration") else plain(policy.__dict__)})
            record["input_hash"] = digest([prepared, plain(self.spec), replicate, policy_identity, "deployment"])
            cached = self._cached_record(record["record_id"], record["input_hash"])
            if cached is not None:
                return cached
            feature = prepared.trial["features"]
            row = {"trial_id": trial_id, "task_id": prepared.trial["task_id"],
                   "checkpoint_id": prepared.trial["checkpoint_id"],
                   "group_id": digest(prepared.trial["group_keys"]),
                   "group_keys": prepared.trial["group_keys"], "x": list(feature["values"]),
                   "feature_names": list(feature["names"]), "feature_version": feature["version"],
                   "checks": prepared.trial["checks"], "eligible": prepared.trial["checks"].get("v") == 1,
                   "has_candidate": prepared.trial["memory"] is not None,
                   "retrieval_score": prepared.trial.get("retrieval_score", 0.0)}
            gate_started = time.perf_counter()
            if not row["eligible"] or not row["has_candidate"]:
                probability = 0.0
            else:
                output = policy.predict_probability([row])
                probability = finite(float(output[0]), "policy reminder probability", minimum=0)
                if probability > 1:
                    raise ProtocolError("policy probability exceeds one")
            gate_costs = empty_costs()
            gate_costs["wall_seconds"] = time.perf_counter() - gate_started
            gate_costs["gate_calls"] = float(row["eligible"] and row["has_candidate"])
            costs = add_costs(prepared.common_costs, gate_costs)
            deployment_prepared = PreparedDecision(prepared.trial, prepared.checkpoint, prepared.draft,
                                                  costs, prepared.preparation_events, prepared.state_fingerprint,
                                                  prepared.environment_config, prepared.actor_config,
                                                  prepared.actor_capabilities, prepared.environment_capabilities)
            z = int(random.Random(self._seed(trial_id, replicate, "deployment_policy")).random() < probability)
            record["decision"] = {"z": z, "probability": probability, "sampling_seed": self._seed(trial_id, replicate, "deployment_policy"),
                                  "policy_id": policy_identity, "eligible": row["eligible"],
                                  "source": "frozen_policy_not_randomized_logging"}
            record["gate_costs"] = gate_costs
            record["gate_implementation"] = "frozen_policy" if gate_costs["gate_calls"] else "no_policy_call"
            record["common_costs"] = costs
            arm = self._arm(env, actor, deployment_prepared, z, replicate)
            record["arms"] = {"remind" if z else "silent": arm}
            record["observed"] = {"z": z, "reward": arm["utility"], "success": arm["success"]}
            record["valid"] = arm["status"] != "error"
            is_new = trial_id in self._new_preparations
            record["preparation_charged_in_collection"] = is_new
            record["collection_costs"] = add_costs(prepared.common_costs if is_new else empty_costs(), add_costs(gate_costs, arm["incremental_costs"]))
            self._new_preparations.discard(trial_id)
            record["paired_regression"] = None
            self._verify_prepared(prepared)
            return self._save(record)
        except Exception as exc:
            record = self._base_record(prepared, replicate, logging_gate=False) if prepared else {"trial": {"trial_id": trial_id}, "replicate": replicate, "protocol": plain(self.spec)}
            record.pop("assignment", None)
            record.update(schema_version="deployment-intervention.v2", record_id=digest([trial_id, replicate, "deployment", policy_id or type(policy).__name__])[:24],
                          policy_name=policy_id or getattr(policy, "name", type(policy).__name__), valid=False,
                          arms={}, error={"type": type(exc).__name__, "message": str(exc)})
            if isinstance(exc, ProviderError):
                record["failed_attempt"] = exc.attempt
            return self._save(record)
    def collect_single(self, env: Environment, actor: Actor, memory: SourceMemory | Callable | None, *, trial_id: str,
                       replicate: int = 0, group_keys: tuple[str, ...] = (), sampling_weight: float = 1.0,
                       reset: bool = False, common_overhead: dict | None = None) -> dict:
        try:
            if reset:
                env.reset(self._seed(trial_id, replicate, "reset"))
            prepared = self.prepare(env, actor, memory, trial_id=trial_id, replicate=replicate,
                                    group_keys=group_keys, sampling_weight=sampling_weight, common_overhead=common_overhead)
            record = self._base_record(prepared, replicate)
            record.update(schema_version="randomized-single.v2", record_id=digest([trial_id, replicate, "single"])[:24],
                          input_hash=digest([prepared, plain(self.spec), replicate, "single"]), valid=False, error=None)
            cached = self._cached_record(record["record_id"], record["input_hash"])
            if cached is not None:
                return cached
            if prepared.trial["memory"] is None or prepared.trial["checks"].get("v") != 1:
                z = 0
                record.update(schema_version="ineligible-decision.v2", exclusion_reasons=["no_eligible_source_memory"])
            else:
                z = record["assignment"]["z"]
            arm = self._arm(env, actor, replace(prepared, common_costs=record["common_costs"]), z, replicate)
            record["arms"] = {"remind" if z else "silent": arm}
            record["valid"] = arm["status"] != "error" and record["schema_version"] == "randomized-single.v2"
            is_new = trial_id in self._new_preparations
            record["preparation_charged_in_collection"] = is_new
            record["collection_costs"] = add_costs(prepared.common_costs if is_new else empty_costs(),
                                                  add_costs(record.get("logging_gate_costs", {}), arm["incremental_costs"]))
            self._new_preparations.discard(trial_id)
            if record["valid"]:
                record["observed"] = {"z": z, "propensity": record["assignment"]["propensity"],
                                      "reward": arm["utility"], "success": arm["success"]}
            return self._save(record)
        except Exception as exc:
            record = {"schema_version": "randomized-single.v2", "record_id": digest([trial_id, replicate, "single"])[:24],
                      "trial": {"trial_id": trial_id}, "protocol": plain(self.spec), "replicate": replicate,
                      "valid": False, "arms": {}, "error": {"type": type(exc).__name__, "message": str(exc)}}
            if isinstance(exc, ProviderError):
                record["failed_attempt"] = exc.attempt
            return self._save(record)


# Descriptive aliases retained in the new package only, not legacy compatibility.
PairedCollector = InterventionCollector
SingleArmCollector = InterventionCollector
