"""Reproducible offline development workflow, with real frozen-policy rollouts.

The factorial suite is an executable plumbing experiment, not evidence for H1--H4.
Every source lineage gets its own bank. Collection freezes splits before outcomes.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

from .config import configuration_fingerprint, load_config
from .contracts import StudySpec, digest, plain, read_json, read_jsonl, stage_seed, write_json, write_jsonl
from .data import build_dataset, build_split_manifest, load_collection
from .environments import FactorialEnvironment
from .features import FeatureBuilder, SentenceTransformerEncoder
from .memory import FixedRetriever, MemoryBank, memory_from_dict
from .providers import OpenAICompatibleProvider, OpenAIConfig, ProviderActor, ScriptedActor
from .runtime import InterventionCollector
from .source import EvidenceSpec, build_memory, evaluate_evidence, replay_source, validate_evidence


def _empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance():
    packages = {}
    for name in ("cave-mem-v2", "numpy", "sentence-transformers", "alfworld"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "implementation_sha256": implementation,
            "packages": packages, "reference_commits": {
                "CAVE-Mem": "591b276e55880023365913b60003a9c6cc2352e6",
                "MemRL": "c1b322ca43de36ddf64c6712f89d0095bfc35ce0"}}


def actor_factory(raw, *, real_actor=False):
    provider = raw.get("provider", {})
    if provider:
        if not real_actor:
            raise ValueError("provider configured: explicitly pass --real-actor to make API calls")
        config = OpenAIConfig(**provider)
        return lambda: ProviderActor(OpenAICompatibleProvider(config))
    if real_actor:
        raise ValueError("--real-actor needs an explicit [provider] configuration")
    return lambda: ScriptedActor(overreaction_rate=.25)


class VersionedSelector:
    def __init__(self, retriever):
        self.retriever = retriever
        self.version = retriever.version

    def __call__(self, context):
        memory, score = self.retriever.retrieve(context)
        return memory, score, copy.deepcopy(self.retriever.last_usage)


class WorkflowFeatures:
    def __init__(self, builder):
        self.builder = builder
        self.version = builder.version

    def __call__(self, context, action, memory, checks, spec, *, retrieval_score=0.):
        features = self.builder.build(context, action, memory, checks,
                                      remaining_calls=spec.max_model_calls, retrieval_score=retrieval_score)
        return features, self.builder.last_usage


def feature_builder(raw, cache_dir):
    settings = dict(raw.get("encoder", {}))
    encoder = None
    if settings:
        # A real, pinned encoder is used when configured. No fabricated embeddings.
        encoder = SentenceTransformerEncoder(cache_dir=Path(cache_dir), **settings)
    return WorkflowFeatures(FeatureBuilder(encoder=encoder))


def build_development_sources(config, output, *, real_actor=False):
    spec, raw = load_config(config)
    output = _empty_output(output)
    groups = raw.get("experiment", {}).get("groups", 30)
    if type(groups) is not int or groups < 20:
        raise ValueError("the five-way development workflow needs at least 20 source groups")
    make_actor = actor_factory(raw, real_actor=real_actor)
    memories, cases, source_reports = [], [], []
    mechanisms = tuple(FactorialEnvironment.MECHANISMS)
    for group in range(groups):
        mechanism = mechanisms[group % len(mechanisms)]
        source_id = f"source-{group:04d}"
        make_env = lambda sid=source_id, mech=mechanism: FactorialEnvironment(
            task_id=sid, mechanism=mech, applicable=True, safe_intent=False, max_steps=spec.max_steps)
        env = make_env()
        env.reset(spec.base_seed)
        checkpoint = env.snapshot()
        env.close()
        evidence_spec = EvidenceSpec(endpoint="terminal_utility", min_confirmation_pairs=16,
                                     max_steps=spec.max_steps, utility_protocol=plain(spec))
        confirmation = replay_source(make_env, make_actor, checkpoint, {"type": "operate"},
                                     {"type": "prepare"}, replicates=32, base_seed=spec.base_seed,
                                     spec=evidence_spec)
        evidence = evaluate_evidence(confirmation, spec=evidence_spec)
        memory = build_memory(source_task_id=source_id, source_trace_id=source_id + ":trace",
                              source_checkpoint_hash=digest(checkpoint),
                              mechanism=f"{mechanism}: operating without required preparation fails",
                              problem_action={"type": "operate"}, correction={"type": "prepare"},
                              positive_conditions=({"field": "requires_preparation", "equals": True},),
                              evidence=evidence)
        checked = validate_evidence(memory)
        if checked["v"] != 1:
            write_json(output / "failed-source-evidence.json", evidence)
            raise ValueError(f"source evidence did not qualify: {source_id}: {checked}")
        memories.append(memory)
        source_reports.append({"group": group, "memory_id": memory.memory_id,
                               "evidence_hash": memory.evidence["evidence_hash"],
                               "confirmation_report_hash": evidence["evidence_hash"], "validation": checked})
        # Four factorial cells, plus two observable-uncertainty cells.
        for applicable, safe, hidden in ((False, False, False), (False, True, False),
                                        (True, False, False), (True, True, False),
                                        (True, False, True), (True, True, True)):
            identity = f"g{group:04d}-r{int(applicable)}-s{int(safe)}-u{int(hidden)}"
            cases.append({"trial_id": identity, "source_group": f"source-group:{group}",
                          "memory_id": memory.memory_id, "sampling_weight": 1.0,
                          "semantic": {"family": "factorial", "mechanism": mechanism,
                                       "template": ("direct", "checklist", "compact")[group % 3]},
                          "environment": {"task_id": identity, "mechanism": mechanism,
                              "template": ("direct", "checklist", "compact")[group % 3],
                              "applicable": applicable, "safe_intent": safe,
                              "hide_applicability": hidden, "max_steps": spec.max_steps}})
        write_jsonl(output / "source-replays" / f"{source_id}.jsonl", confirmation)
    bank = MemoryBank(output / "memories.jsonl", memories)
    suite = {"schema_version": "factorial-suite.v2", "research_role": "fixture_only",
             "study": plain(spec), "config_hash": configuration_fingerprint(spec, raw),
             "bank_hash": bank.content_hash(), "cases": cases,
             "natural_distribution": "declared uniform over the six fixture cells; not real deployment frequencies",
             "source_groups": groups, "sources": source_reports, "runtime": provenance()}
    suite["suite_hash"] = digest(suite)
    write_json(output / "suite.json", suite)
    return suite


def _load_suite(path):
    path = Path(path)
    suite = read_json(path / "suite.json")
    if suite.get("suite_hash") != digest({k: v for k, v in suite.items() if k != "suite_hash"}):
        raise ValueError("suite checksum mismatch")
    bank = MemoryBank(path / "memories.jsonl")
    if bank.content_hash() != suite["bank_hash"]:
        raise ValueError("frozen suite memory bank changed")
    return suite, bank


def collect_development(config, sources, output, *, kind="paired", real_actor=False):
    spec, raw = load_config(config)
    suite, bank = _load_suite(sources)
    if suite["config_hash"] != configuration_fingerprint(spec, raw):
        raise ValueError("suite configuration changed; create a new source/run cohort")
    if kind not in ("paired", "single"):
        raise ValueError("choose paired or single")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_input = {"config_hash": suite["config_hash"], "suite_hash": suite["suite_hash"], "kind": kind}
    marker = output / "collection-input.json"
    if marker.exists() and read_json(marker) != run_input:
        raise ValueError("collection output belongs to a different cohort")
    write_json(marker, run_input)
    make_actor = actor_factory(raw, real_actor=real_actor)
    collector = InterventionCollector(spec, artifact_dir=output,
                                      feature_builder=feature_builder(raw, output / "encoder-cache"))
    prepared = []
    for case in suite["cases"]:
        env = FactorialEnvironment(**case["environment"])
        env.reset(stage_seed(spec.base_seed, case["trial_id"], 0, "reset"))
        selector = VersionedSelector(FixedRetriever([bank.get(case["memory_id"])]))
        decision = collector.prepare(env, make_actor(), selector, trial_id=case["trial_id"],
                                     group_keys=(case["source_group"],),
                                     sampling_weight=case["sampling_weight"])
        prepared.append((case, decision))
        env.close()
    # Outcome-blind manifest is saved before the first target continuation.
    manifest = build_split_manifest([p.trial for _, p in prepared],
                                   seed=raw.get("experiment", {}).get("split_seed", 64),
                                   semantic_metadata={c["trial_id"]: c["semantic"] for c, _ in prepared})
    manifest_path = output / "split-manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("refusing to change an existing frozen split")
    write_json(manifest_path, manifest)
    records = []
    for case, decision in prepared:
        for replicate in range(spec.replicates):
            if kind == "paired":
                record = collector.collect_prepared_pair(decision,
                    lambda case=case: FactorialEnvironment(**case["environment"]), make_actor,
                    replicate=replicate)
            else:
                env = FactorialEnvironment(**case["environment"])
                env.restore(copy.deepcopy(decision.checkpoint))
                selector = VersionedSelector(FixedRetriever([bank.get(case["memory_id"])]))
                record = collector.collect_single(env, make_actor(), selector, trial_id=case["trial_id"],
                                                  replicate=replicate, group_keys=(case["source_group"],),
                                                  sampling_weight=case["sampling_weight"])
                env.close()
            records.append(record)
    summary = {"kind": kind, "planned": len(records), "valid": sum(r["valid"] for r in records),
               "manifest_hash": manifest["manifest_hash"], "runtime": provenance()}
    write_json(output / "collection-summary.json", summary)
    return summary


def export_dataset(collection, output):
    return build_dataset(load_collection(collection), output,
                         manifest=read_json(Path(collection) / "split-manifest.json"))["summary"]


def load_partitions(path):
    path = Path(path)
    sums = read_json(path / "checksums.json")
    for name, expected in sums.items():
        if digest((path / name).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"dataset file checksum mismatch: {name}")
    return {s: read_jsonl(path / f"{s}.jsonl") for s in ("train", "calibration", "selection", "audit", "test")}


def train_models(config, dataset, output):
    from .learning import BinaryPolicy, OutcomeModel, cross_fit_rewards
    spec, raw = load_config(config)
    parts = load_partitions(dataset)
    output = _empty_output(output)
    settings = raw.get("training", {})
    hidden, epochs = settings.get("hidden_size", 16), settings.get("epochs", 300)
    seed, rate = settings.get("seed", 42), settings.get("learning_rate", .01)
    quality = read_json(Path(dataset) / "quality.json")
    if any(r.get("metadata", {}).get("protocol_hash") != digest(spec)
           for rows in parts.values() for r in rows):
        raise ValueError("training config does not match the collected study protocol")
    if quality["kind"] == "paired":
        outcome = OutcomeModel(hidden=hidden, seed=seed).fit(parts["train"], epochs=epochs, learning_rate=rate)
        outcome.calibrate(parts["calibration"], epochs=epochs)
        outcome.save(output / "outcome.json")
        sft = BinaryPolicy(hidden=hidden, seed=seed).warm_start(parts["train"], outcome, epochs=epochs)
        sft.save(output / "supervised.json")
        direct = BinaryPolicy.load(output / "supervised.json")
        direct.fit_paired(parts["train"], epochs=epochs, kl_weight=spec.kl_weight, constrained=False)
        direct.save(output / "direct.json")
        policy = BinaryPolicy.load(output / "supervised.json")
        policy.fit_paired(parts["train"], parts["selection"], epochs=epochs,
                          risk_budget=spec.risk_budget, compute_budget=spec.compute_budget,
                          kl_weight=spec.kl_weight)
        policy.save(output / "policy.json")
        policy.save(output / "constrained-candidate.json")
    else:
        crossfit = cross_fit_rewards(parts["train"], hidden=hidden, seed=seed, epochs=epochs)
        write_json(output / "train-crossfit.json", crossfit)
        policy = BinaryPolicy(hidden=hidden, seed=seed).warm_start_logged(parts["train"], crossfit, epochs=epochs)
        policy.save(output / "supervised.json")
        policy.fit_dr(parts["train"], crossfit, epochs=epochs, kl_weight=spec.kl_weight, constrained=False)
        policy.save(output / "policy.json")
        policy.save(output / "dr-candidate.json")
    report = {"kind": quality["kind"], "dataset_manifest": quality["manifest_hash"],
              "policy_report": policy.report, "runtime": provenance(),
              "selection_note": "paired selection is followed by fresh rollout validation in pipeline",
              "config_hash": configuration_fingerprint(spec, raw)}
    write_json(output / "training.json", report)
    return report


def select_with_rollouts(config, sources, dataset, models, output, *, real_actor=False):
    """Use selection tasks only; pair audits constrain actual rollout selection."""
    from .evaluation import evaluate_paired
    from .learning import BinaryPolicy
    spec, _ = load_config(config)
    models, output = Path(models), _empty_output(output)
    rows = load_partitions(dataset)["selection"]
    kind = read_json(Path(dataset) / "quality.json")["kind"]
    filenames = (["supervised.json", "direct.json", "constrained-candidate.json"]
                 if kind == "paired" else ["supervised.json", "dr-candidate.json"])
    reports, winner, best = {}, None, float("-inf")
    for filename in filenames:
        policy = BinaryPolicy.load(models / filename)
        report = live_evaluate(config, sources, dataset, models, output / Path(filename).stem,
                               split="selection", real_actor=real_actor, policy_file=filename)
        paired = evaluate_paired(policy, rows, allow_selection=True, bootstrap_samples=300) if kind == "paired" else None
        feasible = (paired is None or (paired["metrics"]["regression"]["upper"] <= spec.risk_budget
                    and paired["metrics"]["extra_compute"]["upper"] <= spec.compute_budget))
        value = report["metrics"].get("value", {}).get("estimate")
        reports[filename] = {"actual_rollouts": report, "paired_selection": paired, "feasible": feasible}
        if feasible and value is not None and value > best:
            winner, best = filename, value
    if winner is None:
        winner = "constrained-candidate.json" if kind == "paired" else "supervised.json"
        selected = BinaryPolicy.load(models / winner)
        selected.force_silent = True
    else:
        selected = BinaryPolicy.load(models / winner)
    selected.selection_rows = [{k: r[k] for k in ("group_id", "group_keys", "task_id", "checkpoint_id", "trial_id", "source_disjoint") if k in r} for r in rows]
    selected.report = {**selected.report, "rollout_selection": {"winner": winner,
        "value": best if best != float("-inf") else None, "selection_hash": digest(rows),
        "fallback_silent": best == float("-inf"), "risk_identified": kind == "paired"}}
    selected.save(models / "policy.json")
    summary = {"winner": winner, "actual_selection_value": best if best != float("-inf") else None,
               "fallback_silent": best == float("-inf"), "candidates": reports,
               "paired_harm_identified": kind == "paired",
               "note": "single-arm selection cannot impose the paired regression constraint"}
    write_json(output / "selection.json", summary)
    return summary


def evaluate_models(dataset, models, output, *, split="test", bootstrap_samples=300,
                    research_role="unspecified dataset provenance; estimator diagnostics only"):
    from .baselines import RulePolicy, SupervisedGate
    from .evaluation import evaluate_logged, evaluate_paired, calibration_report
    from .learning import BinaryPolicy, OutcomeModel, cross_fit_rewards
    if split not in ("audit", "test"):
        raise ValueError("final evaluation requires untouched audit/test")
    parts = load_partitions(dataset)
    rows = parts[split]
    kind = read_json(Path(dataset) / "quality.json")["kind"]
    models = Path(models)
    if read_json(models / "training.json")["dataset_manifest"] != read_json(Path(dataset) / "quality.json")["manifest_hash"]:
        raise ValueError("model artifacts belong to a different dataset manifest")
    policies = {"silent": RulePolicy("silent"), "always_remind": RulePolicy("always_remind"),
                "hard_vre": RulePolicy("hard_vre"), "applicability": RulePolicy("applicability"),
                "exposure": RulePolicy("exposure"), "source_only": RulePolicy("source_only"),
                "supervised": BinaryPolicy.load(models / "supervised.json"),
                "learned": BinaryPolicy.load(models / "policy.json")}
    if kind == "paired":
        outcome = OutcomeModel.load(models / "outcome.json")
        policies["calibrated_threshold"] = SupervisedGate(outcome)
        if (models / "constrained-candidate.json").exists():
            policies["constrained"] = BinaryPolicy.load(models / "constrained-candidate.json")
        if (models / "direct.json").exists():
            policies["direct"] = BinaryPolicy.load(models / "direct.json")
        reports = {name: evaluate_paired(p, rows, bootstrap_samples=bootstrap_samples) for name, p in policies.items()}
        calibration = calibration_report(outcome, rows)
    else:
        if (models / "dr-candidate.json").exists():
            policies["dr"] = BinaryPolicy.load(models / "dr-candidate.json")
        crossfit = cross_fit_rewards(rows, folds=len({r["group_id"] for r in rows}), epochs=100)
        reports = {name: evaluate_logged(p, rows, crossfit, bootstrap_samples=bootstrap_samples) for name, p in policies.items()}
        calibration = None
    report = {"kind": kind, "split": split, "policies": reports, "calibration": calibration,
              "role": research_role}
    write_json(output, report)
    return report


def live_evaluate(config, sources, dataset, models, output, *, split="test", real_actor=False, policy_file="policy.json"):
    from .evaluation import evaluate_deployment
    from .learning import BinaryPolicy
    spec, raw = load_config(config)
    suite, bank = _load_suite(sources)
    if suite["config_hash"] != configuration_fingerprint(spec, raw):
        raise ValueError("rollout configuration differs from the frozen suite")
    if read_json(Path(models) / "training.json")["dataset_manifest"] != read_json(Path(dataset) / "quality.json")["manifest_hash"]:
        raise ValueError("rollout model and dataset manifests differ")
    parts = load_partitions(dataset)
    rows = parts[split]
    manifest = read_json(Path(dataset) / "split-manifest.json")
    if policy_file is None:
        from .baselines import RulePolicy
        policy = RulePolicy("silent")
        policy_id = "no-memory-wrapper.v2"
    else:
        policy = BinaryPolicy.load(Path(models) / policy_file)
        policy_id = digest(read_json(Path(models) / policy_file))
    output = _empty_output(output)
    collector = InterventionCollector(spec, artifact_dir=output / "records",
                                      feature_builder=feature_builder(raw, output / "encoder-cache"))
    make_actor = actor_factory(raw, real_actor=real_actor)
    chosen = {r["trial_id"] for r in rows}
    records = []
    for case in suite["cases"]:
        if case["trial_id"] not in chosen:
            continue
        for replicate in range(spec.replicates):
            env = FactorialEnvironment(**case["environment"])
            env.reset(stage_seed(spec.base_seed, case["trial_id"], 0, "reset"))
            selector = None if policy_file is None else VersionedSelector(FixedRetriever([bank.get(case["memory_id"])]))
            record = collector.collect_deployment(env, make_actor(), selector, policy,
                trial_id=case["trial_id"], replicate=replicate, group_keys=(case["source_group"],),
                sampling_weight=case["sampling_weight"], policy_id=policy_id)
            # Group identifiers come from the pre-existing, outcome-blind manifest.
            record["evaluation_group_id"] = manifest["trial_assignments"][case["trial_id"]]["group_id"]
            record["evaluation_split"] = split
            record["population"] = split
            record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
            write_json(output / "annotated" / (record["record_id"] + ".json"), record)
            records.append(record)
            env.close()
    report = evaluate_deployment(records, policy=policy, source_disjoint=manifest["source_disjoint"], allow_selection=split == "selection")
    report["system"] = "no_memory" if policy_file is None else policy_file
    write_json(output / "report.json", report)
    return report


def run_pipeline(config, output, *, kind="paired", real_actor=False):
    output = _empty_output(output)
    build_development_sources(config, output / "sources", real_actor=real_actor)
    collect_development(config, output / "sources", output / "collection", kind=kind, real_actor=real_actor)
    export_dataset(output / "collection", output / "dataset")
    train_models(config, output / "dataset", output / "models")
    # Selection validation executes full tasks with the frozen candidate.
    select_with_rollouts(config, output / "sources", output / "dataset", output / "models",
                         output / "selection-rollouts", real_actor=real_actor)
    evaluate_models(output / "dataset", output / "models", output / "audit-report.json", split="audit",
                    research_role="fixture plumbing only; not empirical evidence for the research hypotheses")
    evaluate_models(output / "dataset", output / "models", output / "test-report.json", split="test",
                    research_role="fixture plumbing only; not empirical evidence for the research hypotheses")
    live_evaluate(config, output / "sources", output / "dataset", output / "models",
                  output / "test-rollout", split="test", real_actor=real_actor)
    live_evaluate(config, output / "sources", output / "dataset", output / "models",
                  output / "no-memory-rollout", split="test", real_actor=real_actor, policy_file=None)
    report = {"status": "complete", "kind": kind, "output": str(output.resolve()),
              "research_role": "fixture_only", "runtime": provenance()}
    write_json(output / "run.json", report)
    return report
