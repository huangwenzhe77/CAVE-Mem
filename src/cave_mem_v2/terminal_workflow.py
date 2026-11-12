"""Real Terminal-Bench tasks under a declared filesystem checkpoint protocol.

Source discovery and source confirmation are separate commands: a reviewed
candidate file freezes repairs before confirmation. Target retrieval searches
the complete frozen source bank using only public pre-intervention context.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

from .config import configuration_fingerprint, load_config
from .contracts import (digest, plain, read_json, read_jsonl, stage_seed, utility,
                        write_json, write_jsonl)
from .data import build_dataset, build_split_manifest, load_collection
from .features import DEFAULT_FACT_NAMES, FeatureBuilder, MechanismAssessor, SentenceTransformerEncoder
from .memory import FixedRetriever, MemoryBank
from .providers import OpenAICompatibleProvider, OpenAIConfig, ProviderActor, ProviderError
from .runtime import InterventionCollector, add_costs, empty_costs, response_costs
from .source import EvidenceSpec, build_memory, evaluate_evidence, locate_candidates, replay_source, validate_evidence
from .terminal import DockerBackend, TerminalEnvironment, load_terminal_tasks
from .workflow import (VersionedSelector, WorkflowFeatures, _empty_output, evaluate_models,
                       load_partitions, provenance, train_models)


TERMINAL_PROMPT = (
    'Complete the public terminal task. Return exactly one JSON action: '
    '{"type":"shell","command":"a bash command"} or {"type":"finish"}. '
    'Each command starts a new shell in the declared work directory. Only filesystem '
    'changes persist; shell variables, working-directory changes and processes do not. '
    'Use explicit paths or combine dependent shell operations in the same command. '
    'The task verifier and its private tests are unavailable to you. '
    'Do not inspect /tests, /solution or /logs/verifier. Finish when the task is complete.'
)

TERMINAL_BASELINES = ("silent", "always_remind", "source_only", "applicability", "exposure", "hard_vre")


def _check_group_counts(split, *, kind="paired"):
    required = {name: 2 for name in ("train", "calibration", "selection", "audit", "test")}
    if kind == "single":
        required["train"] = 3  # The declared training nuisance estimator uses three folds.
    missing = {name: {"actual": split["group_counts"].get(name, 0), "minimum": minimum}
               for name, minimum in required.items() if split["group_counts"].get(name, 0) < minimum}
    if missing:
        raise ValueError("insufficient independent groups for training/evaluation: " + str(missing))
    return split["group_counts"]


def _planned_cohort(raw, identity, *, kind="paired"):
    """Upper bound on group counts before any API call; real source links follow."""
    settings = raw["terminal"]
    task_map = {"terminal-target-" + digest([identity, task])[:24]: task for task in settings["target_ids"]}
    trials = [{"trial_id": trial, "task_id": task, "checkpoint_id": "planned:" + task}
              for trial, task in task_map.items()]
    declared = settings.get("assignments")
    if declared is not None and set(declared) != set(settings["target_ids"]):
        raise ValueError("terminal.assignments must cover exactly target_ids")
    split = build_split_manifest(trials, source_disjoint=settings["source_disjoint"],
        assignments={trial: declared[task] for trial, task in task_map.items()} if declared is not None else None,
        seed=raw.get("experiment", {}).get("split_seed", 42),
        semantic_metadata={trial: settings.get("semantics", {}).get(task, {}) for trial, task in task_map.items()},
        holdout_axis=settings.get("holdout_axis"), holdout_values=settings.get("holdout_values", ()))
    return {"target_group_upper_bound": _check_group_counts(split, kind=kind),
            "source_dependencies_checked_after_retrieval": settings["source_disjoint"],
            "minimum_counts_are_operational_not_statistical_power": True}


def terminal_actor_factory(raw, *, real_actor=False):
    if not real_actor:
        raise ValueError("terminal execution requires --real-actor and an explicitly configured provider")
    if not raw.get("provider"):
        raise ValueError("terminal execution needs [provider]; there is no scripted actor fallback")
    config = OpenAIConfig(**raw["provider"])
    if not os.environ.get(config.api_key_env):
        raise ValueError("required credential environment variable is unset: " + config.api_key_env)
    return lambda: ProviderActor(OpenAICompatibleProvider(config), system_prompt=TERMINAL_PROMPT,
                                  prompt_version="terminal-fs-boundary.v1")


def _setup(config, manifest, *, real_actor=False, make_actor=None, make_env=None):
    spec, raw = load_config(config)
    tasks = {task.task_id: task for task in load_terminal_tasks(manifest)}
    settings = raw.get("terminal", {})
    source_ids, target_ids = settings.get("source_ids", []), settings.get("target_ids", [])
    if not source_ids or not target_ids or set(source_ids) & set(target_ids):
        raise ValueError("declare nonempty, disjoint terminal source_ids and target_ids")
    if len(source_ids) != len(set(source_ids)) or len(target_ids) != len(set(target_ids)):
        raise ValueError("duplicate terminal task identities")
    if (set(source_ids) | set(target_ids)) - set(tasks):
        raise ValueError("terminal config names tasks absent from the frozen manifest")
    if {tasks[k].asset_hash for k in source_ids} & {tasks[k].asset_hash for k in target_ids}:
        raise ValueError("source and target tasks share identical assets under different IDs")
    if type(settings.get("source_disjoint")) is not bool:
        raise ValueError("explicitly declare terminal.source_disjoint; shared-bank studies use false")
    checkpoint_steps = settings.get("checkpoint_steps", 0)
    if type(checkpoint_steps) is not int or checkpoint_steps < 0:
        raise ValueError("checkpoint_steps must be a prespecified nonnegative integer")
    # The environment must score at the same step boundary as the task protocol.
    for task in tasks.values():
        if task.spec["protocol"]["max_steps"] != spec.max_steps + checkpoint_steps:
            raise ValueError("task protocol max_steps must equal study.max_steps + terminal.checkpoint_steps")
    actor = make_actor or terminal_actor_factory(raw, real_actor=real_actor)
    environment = make_env or (lambda task: TerminalEnvironment(task))
    return spec, raw, tasks, actor, environment


def preflight_terminal(manifest, output, *, config=None, backend=None):
    """Read assets, task capability declarations and Docker availability; no API."""
    tasks = load_terminal_tasks(manifest)
    reports = []
    for task in tasks:
        env = TerminalEnvironment(task, backend=backend)
        reports.append({"task": task.public_manifest(), "capabilities": env.capabilities(),
                        "configuration": env.configuration()})
    try:
        docker = (backend or DockerBackend()).doctor()
        docker_ok = True
    except Exception as exc:
        docker, docker_ok = {"error": type(exc).__name__, "message": str(exc)}, False
    report = {"schema_version": "terminal-preflight.v1", "manifest_hash": digest(read_json(manifest)),
              "tasks": reports, "docker_available": docker_ok, "docker": docker,
              "api_called": False, "official_harbor_session_equivalence": False}
    if config:
        spec, raw = load_config(config)
        report["config_hash"] = configuration_fingerprint(spec, raw)
        provider = raw.get("provider", {})
        report["credential_present"] = bool(os.environ.get(provider.get("api_key_env", "OPENAI_API_KEY")))
        try:
            report["planned_cohort"] = _planned_cohort(raw, _identity(spec, raw, manifest))
            report["training_cohort_feasible"] = True
        except (ValueError, KeyError) as exc:
            report["training_cohort_feasible"] = False
            report["planned_cohort_error"] = str(exc)
    write_json(output, report)
    return report


def _identity(spec, raw, manifest):
    return {"config_hash": configuration_fingerprint(spec, raw), "manifest_hash": digest(read_json(manifest))}


def _checked(path, *, key="checksum"):
    value = read_json(path)
    if value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError("artifact checksum mismatch: " + str(path))
    return value


def _write_checked(path, value):
    value = plain(value)
    value["checksum"] = digest(value)
    write_json(path, value)
    return value


def pilot_terminal(config, manifest, output, *, role="source", real_actor=False,
                   make_actor=None, make_env=None):
    """Collect memory-free natural trajectories and private action checkpoints."""
    if role not in ("source", "target"):
        raise ValueError("pilot role must be source or target")
    spec, raw, tasks, actor_factory, env_factory = _setup(config, manifest, real_actor=real_actor,
                                                        make_actor=make_actor, make_env=make_env)
    output = _empty_output(output)
    identity = _identity(spec, raw, manifest)
    traces, candidates = [], []
    for task_id in raw["terminal"][role + "_ids"]:
        task = tasks[task_id]
        trace_id = "terminal-pilot-" + digest([identity, role, task_id])[:24]
        env, actor = env_factory(task), actor_factory()
        costs, steps, status, success, error = empty_costs(), [], "error", None, None
        verifier_reference = None
        started = time.perf_counter()
        try:
            context = env.reset(stage_seed(spec.base_seed, trace_id, 0, "reset"))
            actor_identity = actor.configuration()
            for index in range(spec.max_model_calls):
                checkpoint = copy.deepcopy(env.snapshot())
                response = actor.propose(copy.deepcopy(context), stage="pilot", reminder=None,
                    seed=stage_seed(spec.base_seed, trace_id, 0, "actor", index), call_id=f"{trace_id}:{index}")
                costs = add_costs(costs, response_costs(response))
                if actor.configuration() != actor_identity:
                    raise ValueError("pilot actor configuration changed")
                private_path = "private/" + trace_id + f"-{index}.json"
                _write_checked(output / private_path, {"checkpoint": checkpoint,
                    "checkpoint_hash": digest(checkpoint), "actor_config": actor_identity,
                    "environment_config": env.configuration(), "context": plain(context),
                    "response": plain(response), "step_index": index})
                valid, reason = env.validate_action(response.action)
                item = {"step_index": index, "checkpoint_file": private_path,
                        "action": response.action, "observation": context.observation,
                        "response": plain(response), "valid": valid, "reason": reason}
                steps.append(item)
                if not valid:
                    status, success = "invalid_action", False
                    break
                result = env.step(response.action, stage_seed(spec.base_seed, trace_id, 0, "environment", index))
                costs = add_costs(costs, {"tool_calls": 1., "tool_cost": result.tool_cost})
                item.update(result=plain(result), next_observation=result.context.observation)
                context = result.context
                if result.terminal:
                    status, success = "terminal", result.success
                    break
            else:
                status, success = "budget_exhausted", False
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, ProviderError):
                costs = add_costs(costs, exc.attempt.get("usage", {}))
        finally:
            costs["wall_seconds"] = time.perf_counter() - started
            verifier = getattr(env, "verifier_report", None)
            if verifier is not None:
                verifier_path = "private/" + trace_id + "-verifier.json"
                verifier_artifact = _write_checked(output / verifier_path, {"trace_id": trace_id, "report": verifier})
                verifier_reference = {"path": verifier_path, "checksum": verifier_artifact["checksum"]}
            env.close()
        trace = _write_checked(output / "traces" / (trace_id + ".json"), {
            "schema_version": "terminal-pilot-trace.v1", "trace_id": trace_id,
            "task_id": task_id, "asset_hash": task.asset_hash, "role": role, **identity,
            "status": status, "success": success, "error": error, "costs": costs,
            "private_verifier_reference": verifier_reference,
            "utility": utility(success, costs, spec) if type(success) is bool else None,
            "steps": steps, "research_role": "natural_no_memory_pilot"})
        traces.append({k: trace[k] for k in ("trace_id", "task_id", "status", "success", "checksum")})
        if role == "source" and success is False:
            located = locate_candidates(steps)
            # A terminal failed test can occur despite locally valid actions.
            # Offer an unconfirmed source step for review, never label it causal.
            if not located and steps:
                located = [{"step_index": len(steps)-1, "problem_action": steps[-1]["action"],
                            "reasons": ["terminal_failure_requires_review"]}]
            for candidate in located:
                candidates.append({"trace_id": trace_id, "step_index": candidate["step_index"],
                    "problem_action": candidate["problem_action"], "reasons": candidate["reasons"],
                    "approved": False, "correction": None, "mechanism": "",
                    "positive_conditions": [], "exclusion_conditions": []})
    _write_checked(output / "pilot.json", {"schema_version": "terminal-pilot.v1", **identity,
        "role": role, "traces": traces, "runtime": provenance()})
    # Deliberately editable; confirmation binds the final reviewed contents.
    write_json(output / "candidates.json", {"schema_version": "terminal-source-candidates.v1",
        "pilot_hash": digest(read_json(output / "pilot.json")), "candidates": candidates})
    return {"traces": len(traces), "failed_tasks": sum(t["success"] is False for t in traces),
            "candidate_steps": len(candidates), "next": "review candidates.json before terminal-source"}


def build_terminal_sources(config, manifest, pilot, candidates, output, *, real_actor=False,
                           make_actor=None, make_env=None):
    spec, raw, tasks, actor_factory, env_factory = _setup(config, manifest, real_actor=real_actor,
                                                        make_actor=make_actor, make_env=make_env)
    output, pilot = _empty_output(output), Path(pilot)
    identity = _identity(spec, raw, manifest)
    pilot_info = _checked(pilot / "pilot.json")
    if pilot_info["role"] != "source" or any(pilot_info[k] != v for k, v in identity.items()):
        raise ValueError("source confirmation requires the matching source pilot")
    reviewed = read_json(candidates)
    if reviewed.get("schema_version") != "terminal-source-candidates.v1" or reviewed.get("pilot_hash") != digest(pilot_info):
        raise ValueError("candidate file belongs to a different source pilot")
    chosen = [c for c in reviewed["candidates"] if c.get("approved") is True]
    if not chosen:
        raise ValueError("no reviewed source corrections were approved")
    identities = [(c["trace_id"], c["step_index"]) for c in chosen]
    if len(set(identities)) != len(identities):
        raise ValueError("freeze one repair per source checkpoint before confirmation")
    settings = raw.get("source", {})
    screen_n, confirm_n = settings.get("screening_pairs", 4), settings.get("confirmation_pairs", 32)
    for value in (screen_n, confirm_n):
        if type(value) is not int or value < 1:
            raise ValueError("source screening/confirmation pair counts must be positive integers")
    evidence_spec = EvidenceSpec(max_steps=spec.max_steps, utility_protocol=plain(spec),
        min_confirmation_pairs=settings.get("min_confirmation_pairs", 16),
        confidence=settings.get("confidence", .95),
        acceptance_epsilon=settings.get("acceptance_epsilon", 0.),
        cost_upper_bounds=settings.get("cost_upper_bounds", {}))
    _write_checked(output / "candidate-plan.json", {"candidates": chosen, "source_spec": plain(evidence_spec),
        "screening_pairs": screen_n, "confirmation_pairs": confirm_n, "candidate_selection": "frozen_before_screening"})
    memories, reports = [], []
    source_trace_ids = {t["trace_id"] for t in pilot_info["traces"]}
    for candidate in chosen:
        trace_id, index = candidate["trace_id"], candidate["step_index"]
        if trace_id not in source_trace_ids or type(index) is not int or index < 0:
            raise ValueError("candidate does not name a registered source checkpoint")
        trace = _checked(pilot / "traces" / (trace_id + ".json"))
        if trace["success"] is not False or trace["task_id"] not in raw["terminal"]["source_ids"]:
            raise ValueError("source repair must come from a recorded failed source task")
        original = trace["steps"][index]
        checkpoint = _checked(pilot / original["checkpoint_file"])
        if candidate["problem_action"] != original["action"] or not candidate.get("mechanism") or not isinstance(candidate.get("correction"), dict):
            raise ValueError("reviewed candidate needs the exact original action, a correction and a mechanism hypothesis")
        task = tasks[trace["task_id"]]
        if task.asset_hash != trace["asset_hash"]:
            raise ValueError("source task assets changed after the pilot")
        if actor_factory().configuration() != checkpoint["actor_config"]:
            raise ValueError("source actor differs from the frozen pilot actor")
        arguments = (lambda task=task: env_factory(task), actor_factory, checkpoint["checkpoint"],
                     candidate["problem_action"], candidate["correction"])
        screening = replay_source(*arguments, replicates=screen_n, base_seed=spec.base_seed, cohort="screening", spec=evidence_spec)
        correction_id = digest(candidate)[:24]
        write_jsonl(output / "source-replays" / (correction_id + ".screening.jsonl"), screening)
        if not all(r["valid"] for r in screening) or sum(r["arms"]["corrected"]["score"] - r["arms"]["original"]["score"] for r in screening) <= 0:
            reports.append({"candidate_id": correction_id, "v": 0, "reason": "screening_not_supported"})
            continue
        confirmation = replay_source(*arguments, replicates=confirm_n, base_seed=spec.base_seed, cohort="confirmation", spec=evidence_spec)
        write_jsonl(output / "source-replays" / (correction_id + ".confirmation.jsonl"), confirmation)
        evidence = evaluate_evidence(confirmation, spec=evidence_spec,
                                     screening_ids=tuple(r["sample_id"] for r in screening))
        memory = build_memory(source_task_id=checkpoint["context"]["task_id"], source_trace_id=trace_id,
            source_checkpoint_hash=digest(checkpoint["checkpoint"]), mechanism=candidate["mechanism"],
            problem_action=candidate["problem_action"], correction=candidate["correction"], evidence=evidence,
            positive_conditions=tuple(candidate.get("positive_conditions", [])),
            exclusion_conditions=tuple(candidate.get("exclusion_conditions", [])))
        check = validate_evidence(memory)
        reports.append({"candidate_id": correction_id, "memory_id": memory.memory_id, **check})
        if check["v"] == 1:
            memories.append(memory)
        _write_checked(output / "evidence" / (correction_id + ".json"), {"memory": plain(memory), "validation": check})
    bank = MemoryBank(output / "memories.jsonl", memories)
    report = _write_checked(output / "sources.json", {"schema_version": "terminal-sources.v1", **identity,
        "bank_hash": bank.content_hash(), "candidate_plan_hash": digest(read_json(output / "candidate-plan.json")),
        "source_ids": raw["terminal"]["source_ids"], "qualified_memories": len(memories), "candidates": reports,
        "runtime": provenance()})
    if not memories:
        raise ValueError("no source memory passed independent confirmation; see sources.json")
    return report


def _bank(sources, identity):
    sources = Path(sources)
    report = _checked(sources / "sources.json")
    if report.get("schema_version") != "terminal-sources.v1" or any(report[k] != v for k, v in identity.items()):
        raise ValueError("source bank belongs to a different frozen terminal study")
    bank = MemoryBank(sources / "memories.jsonl")
    if not bank.list_all() or bank.content_hash() != report["bank_hash"]:
        raise ValueError("source memory bank is empty or changed")
    if any(validate_evidence(m)["v"] != 1 for m in bank.list_all()):
        raise ValueError("source bank includes unqualified evidence")
    return bank


def _components(raw, bank, output, tasks):
    settings = dict(raw.get("encoder", {}))
    encoder = SentenceTransformerEncoder(cache_dir=Path(output) / "encoder-cache", **settings) if settings else None
    fact_names, exposure_rules = set(DEFAULT_FACT_NAMES), {}
    fact_protocols = {}
    for task in tasks.values():
        protocol = task.spec["protocol"]
        fact_names.update(protocol.get("public_fact_rules", {}))
        fact_protocols[task.task_id] = protocol.get("public_fact_rules", {})
        for key, rule in protocol.get("exposure_rules", {}).items():
            if key in exposure_rules and exposure_rules[key] != rule:
                raise ValueError("conflicting frozen exposure rules for memory/mechanism: " + key)
            exposure_rules[key] = rule
    assessor = MechanismAssessor(exposure_rules, allowed_fact_names=fact_names)
    builder = FeatureBuilder(encoder=encoder, allowed_fact_names=fact_names)
    # Bind rule semantics as well as vector names/dimensions to the learned schema.
    builder.version += ":terminal-rules:" + digest([assessor.version, fact_protocols])[:20]
    features = WorkflowFeatures(builder)
    selector = VersionedSelector(FixedRetriever(bank, encoder=encoder,
        min_score=raw.get("retrieval", {}).get("min_score")))
    return features, selector, assessor


def _advance(env, actor, steps, spec, trial_id):
    """Reach a prespecified checkpoint naturally, before retrieval or outcomes."""
    context = env.reset(stage_seed(spec.base_seed, trial_id, 0, "reset"))
    costs = empty_costs()
    for index in range(steps):
        if env.terminal_status()[0]:
            raise ValueError("task terminated before the preregistered checkpoint")
        response = actor.propose(copy.deepcopy(context), stage="prefix", reminder=None,
            seed=stage_seed(spec.base_seed, trial_id, 0, "prefix", index), call_id=f"{trial_id}:prefix:{index}")
        costs = add_costs(costs, response_costs(response))
        valid, reason = env.validate_action(response.action)
        if not valid:
            raise ValueError("invalid prefix action: " + str(reason))
        result = env.step(response.action, stage_seed(spec.base_seed, trial_id, 0, "prefix_environment", index))
        costs = add_costs(costs, {"tool_calls": 1., "tool_cost": result.tool_cost})
        if not result.valid:
            raise ValueError("invalid prefix transition: " + str(result.error))
        context = result.context
    if env.terminal_status()[0]:
        raise ValueError("task terminated at the preregistered checkpoint")
    return costs


def collect_terminal(config, manifest, sources, output, *, kind="paired", real_actor=False,
                     make_actor=None, make_env=None):
    if kind not in ("paired", "single"):
        raise ValueError("collection kind must be paired or single")
    spec, raw, tasks, actor_factory, env_factory = _setup(config, manifest, real_actor=real_actor,
                                                        make_actor=make_actor, make_env=make_env)
    output = _empty_output(output)
    identity = _identity(spec, raw, manifest)
    bank = _bank(sources, identity)
    planned_cohort = _planned_cohort(raw, identity, kind=kind)
    features, selector, assessor = _components(raw, bank, output, tasks)
    collector = InterventionCollector(spec, artifact_dir=output, feature_builder=features, check_builder=assessor)
    prepared, rejected, task_map, prefix_costs = [], [], {}, {}
    for task_id in raw["terminal"]["target_ids"]:
        task, actor = tasks[task_id], actor_factory()
        env = env_factory(task)
        trial_id = "terminal-target-" + digest([identity, task_id])[:24]
        try:
            if kind == "paired" and not env.capabilities().get("restore_certified"):
                raise ValueError("paired target task has no validated filesystem restore scope")
            prefix = _advance(env, actor, raw["terminal"].get("checkpoint_steps", 0), spec, trial_id)
            weight = raw["terminal"].get("sampling_weights", {}).get(task_id, 1.)
            decision = collector.prepare(env, actor, selector, trial_id=trial_id, sampling_weight=weight)
            prepared.append((task, decision))
            task_map[trial_id], prefix_costs[trial_id] = task_id, prefix
        except Exception as exc:
            rejected.append({"task_id": task_id, "trial_id": trial_id,
                             "reason": type(exc).__name__ + ": " + str(exc)})
        finally:
            env.close()
    write_jsonl(output / "preparation-rejected.jsonl", rejected)
    if not prepared:
        raise ValueError("no target checkpoint prepared; inspect preparation-rejected.jsonl")
    settings = raw["terminal"]
    declared = settings.get("assignments")
    assignments = {trial: declared[task] for trial, task in task_map.items()} if declared else None
    semantic = {trial: settings.get("semantics", {}).get(task, {}) for trial, task in task_map.items()}
    split = build_split_manifest([p.trial for _, p in prepared], assignments=assignments,
        source_disjoint=settings["source_disjoint"], seed=raw.get("experiment", {}).get("split_seed", 42),
        semantic_metadata=semantic, holdout_axis=settings.get("holdout_axis"), holdout_values=settings.get("holdout_values", ()))
    write_json(output / "split-manifest.json", split)
    _check_group_counts(split, kind=kind)
    _write_checked(output / "terminal-collection.json", {"schema_version": "terminal-collection.v1", **identity,
        "kind": kind, "bank_hash": bank.content_hash(), "task_map": task_map,
        "prefix_costs": prefix_costs, "prefix_cost_role": "shared_sunk_cost_reported_separately_from_checkpoint_utility",
        "source_disjoint": settings["source_disjoint"], "split_hash": split["manifest_hash"]})
    records = []
    for task, decision in prepared:
        for replicate in range(spec.replicates):
            if kind == "paired":
                record = collector.collect_prepared_pair(decision, lambda task=task: env_factory(task), actor_factory, replicate=replicate)
            else:
                env = env_factory(task)
                try:
                    env.restore(copy.deepcopy(decision.checkpoint))
                    record = collector.collect_single(env, actor_factory(), selector,
                        trial_id=decision.trial["trial_id"], replicate=replicate,
                        sampling_weight=decision.trial["sampling_weight"])
                finally:
                    env.close()
            records.append(record)
    summary = {"kind": kind, "planned_tasks": len(settings["target_ids"]), "prepared_tasks": len(prepared),
        "planned": len(records), "valid": sum(r["valid"] for r in records), "preparation_rejected": len(rejected),
        "manifest_hash": split["manifest_hash"], "source_disjoint": settings["source_disjoint"], "runtime": provenance()}
    summary["planned_cohort"] = planned_cohort
    write_json(output / "collection-summary.json", summary)
    return summary


def rollout_terminal(config, manifest, sources, collection, dataset, models, output, *, split="test",
                     policy_file="policy.json", baseline=None, relevance_threshold=0.,
                     real_actor=False, make_actor=None, make_env=None):
    from .baselines import RulePolicy
    from .evaluation import evaluate_deployment
    from .learning import BinaryPolicy
    if split not in ("selection", "audit", "test"):
        raise ValueError("rollout requires selection/audit/test")
    if baseline is not None and policy_file is None:
        raise ValueError("choose either a retrieval baseline or the no-memory system")
    spec, raw, tasks, actor_factory, env_factory = _setup(config, manifest, real_actor=real_actor,
                                                        make_actor=make_actor, make_env=make_env)
    identity = _identity(spec, raw, manifest)
    bank = _bank(sources, identity)
    cohort = _checked(Path(collection) / "terminal-collection.json")
    split_manifest = read_json(Path(dataset) / "split-manifest.json")
    if any(cohort[k] != v for k, v in identity.items()) or cohort["bank_hash"] != bank.content_hash() or cohort["split_hash"] != split_manifest["manifest_hash"]:
        raise ValueError("rollout study/bank/dataset does not match the collected cohort")
    if read_json(Path(models) / "training.json")["dataset_manifest"] != split_manifest["manifest_hash"]:
        raise ValueError("model was trained on a different dataset manifest")
    rows = load_partitions(dataset)[split]
    chosen = {r["trial_id"] for r in rows}
    output = _empty_output(output)
    if policy_file is None:
        policy, policy_id = RulePolicy("silent"), "terminal-no-memory.v1"
        features, selector, assessor = None, None, None
    elif baseline is not None:
        policy = RulePolicy(baseline, relevance_threshold=relevance_threshold)
        policy_id = digest({"baseline": plain(policy), "bank_hash": bank.content_hash(), **identity})
        features, selector, assessor = _components(raw, bank, output, tasks)
    else:
        policy = BinaryPolicy.load(Path(models) / policy_file)
        policy_id = digest(read_json(Path(models) / policy_file))
        features, selector, assessor = _components(raw, bank, output, tasks)
    collector = InterventionCollector(spec, artifact_dir=output / "records", feature_builder=features, check_builder=assessor)
    records, prefix_ledger = [], []

    def failed_record(trial_id, task, replicate, exc, stage):
        record = {"schema_version": "deployment-intervention.v2",
            "record_id": digest([trial_id, replicate, "deployment", policy_id])[:24],
            "trial": {"trial_id": trial_id, "task_id": task.task_id}, "replicate": replicate,
            "protocol": plain(spec), "valid": False, "arms": {}, "error_stage": stage,
            "error": {"type": type(exc).__name__, "message": str(exc)}}
        if isinstance(exc, ProviderError):
            record["failed_attempt"] = plain(exc.attempt)
        return record

    def retain_record(record, trial_id):
        record["evaluation_group_id"] = split_manifest["trial_assignments"][trial_id]["group_id"]
        record["evaluation_split"] = split
        record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
        write_json(output / "annotated" / (record["record_id"] + ".json"), record)
        records.append(record)

    for trial_id in sorted(chosen):
        task = tasks[cohort["task_map"][trial_id]]
        weight = next(r["sampling_weight"] for r in rows if r["trial_id"] == trial_id)
        initial_env = None
        try:
            initial_env, initial_actor = env_factory(task), actor_factory()
            prefix = _advance(initial_env, initial_actor, raw["terminal"].get("checkpoint_steps", 0), spec, trial_id)
            checkpoint = copy.deepcopy(initial_env.snapshot())
        except Exception as exc:
            for replicate in range(spec.replicates):
                retain_record(failed_record(trial_id, task, replicate, exc, "prefix"), trial_id)
            prefix_ledger.append({"trial_id": trial_id, "costs": None, "status": "prefix_error",
                                  "cost_coverage": "unknown_partial_execution", "physical_prefix_charged": True})
            continue
        finally:
            if initial_env is not None:
                initial_env.close()
        for replicate in range(spec.replicates):
            env = None
            try:
                env, actor = env_factory(task), actor_factory()
                env.restore(copy.deepcopy(checkpoint))
                record = collector.collect_deployment(env, actor, selector, policy, trial_id=trial_id,
                    replicate=replicate, sampling_weight=weight, policy_id=policy_id)
            except Exception as exc:
                record = failed_record(trial_id, task, replicate, exc, "restore_or_setup")
            finally:
                if env is not None:
                    env.close()
            retain_record(record, trial_id)
            prefix_ledger.append({"trial_id": trial_id, "replicate": replicate, "costs": prefix,
                                  "status": "complete", "physical_prefix_charged": replicate == 0})
    write_jsonl(output / "prefix-costs.jsonl", prefix_ledger)
    report = evaluate_deployment(records, policy, source_disjoint=cohort["source_disjoint"], allow_selection=split == "selection")
    report.update(system=baseline or ("no_memory" if policy_file is None else policy_file),
                  checkpoint_steps=raw["terminal"].get("checkpoint_steps", 0),
                  return_scope="checkpoint_continuation; prefix costs reported separately",
                  benchmark="terminal-bench@2.0", official_harbor_session_equivalence=False)
    write_json(output / "report.json", report)
    return report


def select_terminal(config, manifest, sources, collection, dataset, models, output, *, real_actor=False,
                    make_actor=None, make_env=None):
    from .evaluation import evaluate_paired
    from .learning import BinaryPolicy, _lineage
    spec, _ = load_config(config)
    rows, models, output = load_partitions(dataset)["selection"], Path(models), _empty_output(output)
    kind = read_json(Path(dataset) / "quality.json")["kind"]
    filenames = ["supervised.json", "direct.json", "constrained-candidate.json"] if kind == "paired" else ["supervised.json", "dr-candidate.json"]
    reports, winner, best = {}, None, float("-inf")
    for filename in filenames:
        policy = BinaryPolicy.load(models / filename)
        actual = rollout_terminal(config, manifest, sources, collection, dataset, models, output / Path(filename).stem,
            split="selection", policy_file=filename, real_actor=real_actor, make_actor=make_actor, make_env=make_env)
        paired = evaluate_paired(policy, rows, allow_selection=True) if kind == "paired" else None
        complete_rollout = actual["completed"] == actual["planned"] and not actual["excluded"]
        feasible = complete_rollout and (paired is None or (
            paired["metrics"]["regression"]["upper"] <= spec.risk_budget and
            paired["metrics"]["extra_compute"]["upper"] <= spec.compute_budget))
        value = actual["metrics"].get("value", {}).get("estimate")
        reports[filename] = {"actual_rollouts": actual, "paired_selection": paired, "feasible": feasible}
        if feasible and value is not None and value > best:
            best, winner = value, filename
    selected = BinaryPolicy.load(models / (winner or "supervised.json"))
    if winner is None:
        selected.force_silent = True
    selected.selection_rows = _lineage(rows)
    selected.report["rollout_selection"] = {"winner": winner, "fallback_silent": winner is None,
        "selection_hash": digest(rows), "risk_identified": kind == "paired"}
    selected.save(models / "policy.json")
    report = {"winner": winner, "fallback_silent": winner is None, "candidates": reports,
              "paired_harm_identified": kind == "paired", "actual_selection_value": best if winner else None}
    write_json(output / "selection.json", report)
    return report


def run_terminal_pipeline(config, manifest, sources, output, *, kind="paired", real_actor=False,
                          make_actor=None, make_env=None):
    output = _empty_output(output)
    kwargs = {"real_actor": real_actor, "make_actor": make_actor, "make_env": make_env}
    collect_terminal(config, manifest, sources, output / "collection", kind=kind, **kwargs)
    build_dataset(load_collection(output / "collection"), output / "dataset",
                  manifest=read_json(output / "collection" / "split-manifest.json"), kind=kind)
    train_models(config, output / "dataset", output / "models")
    select_terminal(config, manifest, sources, output / "collection", output / "dataset", output / "models",
                    output / "selection-rollouts", **kwargs)
    for split in ("audit", "test"):
        evaluate_models(output / "dataset", output / "models", output / (split + "-report.json"), split=split,
            research_role="injected terminal workflow validation" if make_env else "terminal-bench filesystem protocol; official Harbor equivalence is not claimed")
    rollout_terminal(config, manifest, sources, output / "collection", output / "dataset", output / "models",
                     output / "test-rollout", **kwargs)
    rollout_terminal(config, manifest, sources, output / "collection", output / "dataset", output / "models",
                     output / "no-memory-rollout", policy_file=None, **kwargs)
    for baseline in TERMINAL_BASELINES:
        rollout_terminal(config, manifest, sources, output / "collection", output / "dataset", output / "models",
                         output / "baseline-rollouts" / baseline, baseline=baseline, **kwargs)
    report = {"status": "complete", "kind": kind, "benchmark": "terminal-bench@2.0",
        "official_harbor_session_equivalence": False, "source_bank": str(Path(sources).resolve()),
        "component_matched_baselines": list(TERMINAL_BASELINES),
        "runtime": provenance(), "manifest_hash": digest(read_json(manifest)),
        "execution_backend": "injected_test_backend" if make_env else "docker", "actor": "injected" if make_actor else "real_provider"}
    write_json(output / "run.json", report)
    return report
