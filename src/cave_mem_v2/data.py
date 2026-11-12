"""Outcome-blind dependency splits and audited, numerical training exports.

Raw collection records remain audit artifacts. Only the explicit numerical
feature contract and recomputed labels cross into learning. Source-disjoint
and shared-bank target-task experiments are distinct, recorded designs.
"""
from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .contracts import (FeatureRecord, StudySpec, digest, finite, plain, read_json,
                        read_jsonl, truth, utility, weighted_cost, write_json, write_jsonl)
from .memory import memory_from_dict
from .source import validity

SPLITS = ("train", "calibration", "selection", "audit", "test")
DEFAULT_RATIOS = {"train": .5, "calibration": .15, "selection": .15, "audit": .1, "test": .1}
SCHEMAS = {"paired-intervention.v2": "paired", "randomized-single.v2": "single"}


def _trial(value: dict) -> dict:
    value = plain(value)
    trial = value.get("trial", value)
    if not isinstance(trial, dict):
        raise ValueError("trial must be a mapping")
    for key in ("trial_id", "task_id", "checkpoint_id"):
        if not isinstance(trial.get(key), str) or not trial[key]:
            raise ValueError(f"missing pre-treatment {key}")
    return trial


def _dependency_keys(trial: dict, source_disjoint: bool) -> set[str]:
    keys = {"trial:" + trial["trial_id"], "task:" + trial["task_id"],
            "checkpoint:" + trial["checkpoint_id"]}
    for key in trial.get("group_keys", []):
        if not isinstance(key, str) or not key:
            raise ValueError("dependency group keys must be nonempty strings")
        if not source_disjoint and key.startswith(("source_task:", "source_trace:", "memory:")):
            continue
        # Runtime emits the source task both as task:X and source_task:X.
        if not source_disjoint and trial.get("memory") and key == "task:" + trial["memory"]["source_task_id"]:
            continue
        keys.add(key)
    if source_disjoint and trial.get("memory"):
        memory = memory_from_dict(trial["memory"])
        keys.update(("task:" + memory.source_task_id, "source_task:" + memory.source_task_id,
                     "source_trace:" + memory.source_trace_id, "memory:" + memory.content_hash()))
    return keys


def _ratios(value: dict | None) -> dict[str, float]:
    ratios = dict(DEFAULT_RATIOS if value is None else value)
    if not ratios or set(ratios) - set(SPLITS):
        raise ValueError("unknown or empty split ratios")
    ratios = {k: finite(v, "split ratio", minimum=0) for k, v in ratios.items()}
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("split ratios must sum to one")
    return {k: ratios[k] for k in SPLITS if ratios.get(k, 0) > 0}


def build_split_manifest(trials: Iterable[dict], *, ratios: dict | None = None, seed: int = 42,
                         source_disjoint: bool = True, semantic_metadata: dict | None = None,
                         holdout_axis: str | None = None, holdout_values: Iterable[str] = (),
                         assignments: dict[str, str] | None = None,
                         frozen_before_collection: bool = True) -> dict:
    """Freeze linked tasks/checkpoints/source lineages without reading outcomes.

    Call this on prepared trials before collection for a preregistered manifest.
    A semantic holdout is one explicitly chosen axis, never all metadata fields.
    When holdout_values are given they alone form test; a dependency component
    crossing those values and other values is an error, not a relaxed split.
    Optional assignments pin every trial and must respect all dependencies.
    """
    if type(seed) is not int or type(source_disjoint) is not bool:
        raise ValueError("invalid seed/source-disjoint design")
    if type(frozen_before_collection) is not bool:
        raise ValueError("frozen_before_collection must be bool")
    ratios = _ratios(ratios)
    semantic_metadata = copy.deepcopy(semantic_metadata or {})
    if holdout_axis not in (None, "family", "mechanism", "template"):
        raise ValueError("choose one declared family/mechanism/template holdout axis")
    holdout_values = tuple(sorted(set(holdout_values)))
    if holdout_values and holdout_axis is None:
        raise ValueError("holdout values require a semantic axis")
    unique = {}
    for value in trials:
        trial = _trial(value)
        prior = unique.setdefault(trial["trial_id"], trial)
        if digest(prior) != digest(trial):
            raise ValueError("one trial ID has different frozen pre-treatment records")
    if not unique:
        raise ValueError("cannot split an empty trial cohort")
    if set(semantic_metadata) - set(unique):
        raise ValueError("semantic metadata contains an unknown trial")
    parent = {key: key for key in unique}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    owner = {}
    dependencies = {}
    for trial_id in sorted(unique):
        keys = _dependency_keys(unique[trial_id], source_disjoint)
        if holdout_axis:
            value = semantic_metadata.get(trial_id, {}).get(holdout_axis)
            if not isinstance(value, str) or not value:
                raise ValueError("every trial needs the registered holdout semantic value")
            keys.add("semantic:" + holdout_axis + ":" + value)
        dependencies[trial_id] = sorted(keys)
        for key in keys:
            if key in owner:
                union(trial_id, owner[key])
            else:
                owner[key] = trial_id
    groups = defaultdict(list)
    for trial_id in sorted(unique):
        groups[find(trial_id)].append(trial_id)
    components = {"group-" + digest(ids)[:20]: ids for ids in groups.values()}
    assigned = {}
    if assignments is not None:
        if set(assignments) != set(unique):
            raise ValueError("explicit assignments must cover exactly every trial")
        for group, ids in components.items():
            values = {assignments[i] for i in ids}
            if len(values) != 1 or not values <= set(ratios):
                raise ValueError("explicit split conflicts with a dependency component")
            assigned[group] = next(iter(values))
    else:
        forced_test = set()
        if holdout_values:
            if "test" not in ratios:
                raise ValueError("semantic holdout requires a nonzero test allocation")
            present = {semantic_metadata[i][holdout_axis] for i in unique}
            if not set(holdout_values) <= present:
                raise ValueError("registered holdout category is absent")
            for group, ids in components.items():
                selected = {semantic_metadata[i][holdout_axis] in holdout_values for i in ids}
                if len(selected) > 1:
                    raise ValueError("source/task dependency component crosses the semantic holdout boundary")
                if selected == {True}:
                    forced_test.add(group)
                    assigned[group] = "test"
        pool = [g for g in components if g not in forced_test]
        active = {k: v for k, v in ratios.items() if not (holdout_values and k == "test")}
        if len(pool) < len(active):
            raise ValueError(f"insufficient independent groups: {len(pool)} groups for {len(active)} splits")
        if holdout_values and not forced_test:
            raise ValueError("no group in the held-out test categories")
        normalizer = sum(active.values())
        targets = {k: len(pool) * v / normalizer for k, v in active.items()}
        counts = {k: max(1, int(targets[k])) for k in active}
        while sum(counts.values()) > len(pool):
            eligible = [k for k in active if counts[k] > 1]
            key = max(eligible, key=lambda k: (counts[k] - targets[k], k))
            counts[key] -= 1
        while sum(counts.values()) < len(pool):
            key = max(active, key=lambda k: (targets[k] - counts[k], k))
            counts[key] += 1
        ordered = sorted(pool, key=lambda group: digest([seed, group]))
        offset = 0
        for split in SPLITS:
            for group in ordered[offset:offset + counts.get(split, 0)]:
                assigned[group] = split
            offset += counts.get(split, 0)
    if set(assigned.values()) != set(ratios):
        raise ValueError("every requested split needs at least one independent group")
    if holdout_values:
        for group, ids in components.items():
            if any((semantic_metadata[i][holdout_axis] in holdout_values) != (assigned[group] == "test") for i in ids):
                raise ValueError("explicit assignments violate the registered semantic holdout")
    mapping = {trial_id: {"split": assigned[group], "group_id": group,
                          "trial_hash": digest(unique[trial_id])}
               for group, ids in components.items() for trial_id in ids}
    result = {"schema_version": "dependency-split.v2", "seed": seed, "ratios": ratios,
              "source_disjoint": source_disjoint, "frozen_before_collection": frozen_before_collection,
              "holdout_axis": holdout_axis, "holdout_values": list(holdout_values),
              "semantic_metadata": semantic_metadata, "trial_assignments": mapping,
              "dependencies": dependencies, "group_counts": dict(Counter(assigned.values())),
              "trial_counts": dict(Counter(v["split"] for v in mapping.values())),
              "outcome_blind": True}
    result["manifest_hash"] = digest(result)
    return result


def _close(actual, expected, name):
    if actual is None or not math.isclose(finite(actual, name), expected, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(name + " does not match recomputed outcome/cost")


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != "dependency-split.v2" or manifest.get("outcome_blind") is not True:
        raise ValueError("unsupported_split_manifest")
    if manifest.get("manifest_hash") != digest({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("split_manifest_checksum_mismatch")
    assignments, dependencies = manifest["trial_assignments"], manifest["dependencies"]
    if set(assignments) != set(dependencies):
        raise ValueError("split_manifest_dependency_coverage_mismatch")
    owner, group_splits = {}, {}
    for trial_id, assignment in assignments.items():
        group, split = assignment["group_id"], assignment["split"]
        if split not in manifest["ratios"]:
            raise ValueError("unregistered_split_in_manifest")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("one_dependency_group_spans_splits")
        group_splits[group] = split
        for key in dependencies[trial_id]:
            if key in owner and owner[key] != (group, split):
                raise ValueError("shared_dependency_spans_groups_or_splits")
            owner[key] = (group, split)


def record_to_row(record: dict, manifest: dict) -> dict:
    """Validate one completed record, recompute labels and strip raw audit data."""
    if record.get("record_hash") != digest({k: v for k, v in record.items() if k != "record_hash"}):
        raise ValueError("collection_record_checksum_mismatch")
    kind = SCHEMAS.get(record.get("schema_version"))
    if kind is None:
        raise ValueError("unsupported_or_ineligible_record_schema")
    if record.get("valid") is not True:
        raise ValueError("invalid_collection:" + ",".join(record.get("exclusion_reasons", [])))
    trial = _trial(record)
    assignment = manifest["trial_assignments"].get(trial["trial_id"])
    if assignment is None or assignment["trial_hash"] != digest(trial):
        raise ValueError("record_not_in_frozen_split_manifest")
    if not _dependency_keys(trial, manifest["source_disjoint"]) <= set(manifest["dependencies"][trial["trial_id"]]):
        raise ValueError("frozen_split_manifest_omits_required_dependencies")
    features = FeatureRecord(**trial["features"])
    if record.get("feature_hash") != digest(features):
        raise ValueError("frozen_feature_checksum_mismatch")
    checks = {key: truth(trial["checks"][key]) for key in ("v", "rho", "e")}
    memory = memory_from_dict(trial["memory"]) if trial.get("memory") else None
    if checks["v"] != 1 or memory is None or validity(memory) != 1:
        raise ValueError("no_validated_eligible_source_memory")
    spec = StudySpec(**record["protocol"])
    c0 = weighted_cost(record["common_costs"], spec.inference_weights)
    if c0 is None:
        raise ValueError("unknown_unavoidable_inference_overhead")
    sampling_weight = finite(trial.get("sampling_weight", 1), "sampling_weight", minimum=0)
    if sampling_weight <= 0:
        raise ValueError("sampling_weight_must_be_positive")
    replicate = record.get("replicate")
    if type(replicate) is not int or not 0 <= replicate < spec.replicates:
        raise ValueError("invalid_replicate")
    row = {"schema_version": "learner-" + kind + ".v2", "record_id": record["record_id"],
           "trial_id": trial["trial_id"], "task_id": trial["task_id"],
           "checkpoint_id": trial["checkpoint_id"], "group_id": assignment["group_id"],
           "group_keys": manifest["dependencies"][trial["trial_id"]],
           "split": assignment["split"], "replicate": replicate,
           "x": list(features.values), "feature_names": list(features.names),
           "feature_version": features.version, "checks": checks, "eligible": True,
           "sampling_weight": sampling_weight, "c0": c0,
           "source_disjoint": manifest["source_disjoint"],
           "metadata": {"study_id": spec.study_id, "protocol_hash": digest(spec),
                        "actor_hash": digest(record.get("actor_config")),
                        "environment_hash": digest(record.get("environment_config")),
                        "source_evidence_hash": memory.evidence.get("evidence_hash"),
                        "budget_mode": spec.budget_mode, "collection_hash": record["record_hash"]}}

    def arm(name):
        value = record["arms"][name]
        if value["status"] == "error" or type(value["success"]) is not bool:
            raise ValueError("arm_does_not_have_a_valid_binary_terminal_outcome")
        reward = utility(value["success"], value["costs"], spec)
        cost = weighted_cost(value["costs"], spec.inference_weights)
        if reward is None or cost is None:
            raise ValueError("unknown_required_arm_cost")
        _close(value.get("utility"), reward, "arm utility")
        return int(value["success"]), reward, cost

    if kind == "paired":
        if set(record["arms"]) != {"silent", "remind"}:
            raise ValueError("paired_record_needs_exactly_two_arms")
        y0, r0, cost0 = arm("silent")
        y1, r1, cost1 = arm("remind")
        D, H, delta = r1 - r0, int(y0 == 1 and y1 == 0), cost1 - cost0
        for name, expected in (("advantage", D), ("regression", H), ("incremental_cost", delta),
                               ("rescue", int(y0 == 0 and y1 == 1)), ("delta_success", y1-y0)):
            _close(record["labels"].get(name), expected, name)
        row.update(D=D, H=H, delta_cost=delta, y0=y0, y1=y1, r0=r0, r1=r1,
                   cost0=cost0, cost1=cost1)
    else:
        observed, randomized = record["observed"], record["assignment"]
        z = randomized.get("z")
        p = finite(randomized["reminder_probability"], "reminder probability")
        if type(z) is not int or z not in (0, 1) or not 0 < p < 1:
            raise ValueError("invalid_randomized_assignment")
        b = p if z else 1-p
        _close(randomized.get("propensity"), b, "logged propensity")
        _close(observed.get("propensity"), b, "observed propensity")
        if observed.get("z") != z or set(record["arms"]) != {"remind" if z else "silent"}:
            raise ValueError("randomized_assignment_does_not_match_observed_arm")
        y, reward, cost = arm("remind" if z else "silent")
        _close(observed.get("reward"), reward, "observed reward")
        if type(observed.get("success")) is not bool or observed["success"] != bool(y):
            raise ValueError("observed success mismatch")
        row.update(z=z, b=b, r=reward, y=y, cost=cost)
    return row


def build_dataset(records: Iterable[dict], output_dir: str | Path | None = None, *,
                  manifest: dict | None = None, kind: str | None = None, **split_options) -> dict:
    """Export one paired or single cohort, retaining every exclusion reason.

    A manifest made from already collected records is marked retrospective.
    Mixed paired/single records require separate builds and are never merged.
    """
    records = [copy.deepcopy(plain(r)) for r in records]
    if not records:
        raise ValueError("cannot build an empty dataset")
    observed_kinds = {SCHEMAS[r["schema_version"]] for r in records if r.get("schema_version") in SCHEMAS}
    if kind is None:
        if len(observed_kinds) != 1:
            raise ValueError("build paired and randomized-single datasets separately")
        kind = next(iter(observed_kinds))
    if kind not in ("paired", "single"):
        raise ValueError("kind must be paired or single")
    if manifest is None:
        split_options["frozen_before_collection"] = False
        usable_trials = []
        for record in records:
            try:
                usable_trials.append(_trial(record))
            except (ValueError, KeyError, TypeError):
                pass
        manifest = build_split_manifest(usable_trials, **split_options)
    elif split_options:
        raise ValueError("split options cannot modify an existing frozen manifest")
    validate_manifest(manifest)
    identities = Counter((r.get("trial", {}).get("trial_id"), r.get("replicate")) for r in records)
    partitions = {split: [] for split in manifest["ratios"]}
    rejected = []
    for record in records:
        try:
            identity = (record.get("trial", {}).get("trial_id"), record.get("replicate"))
            if identities[identity] != 1:
                raise ValueError("duplicate_trial_replicate")
            if SCHEMAS.get(record.get("schema_version")) != kind:
                raise ValueError("wrong_cohort_or_ineligible_schema")
            row = record_to_row(record, manifest)
            partitions[row["split"]].append(row)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"record_id": record.get("record_id"),
                             "trial_id": record.get("trial", {}).get("trial_id"),
                             "replicate": record.get("replicate"), "reason": str(exc),
                             "collection_hash": record.get("record_hash")})
    schema = set()
    for rows in partitions.values():
        for row in rows:
            schema.add((tuple(row["feature_names"]), row["feature_version"]))
    if len(schema) > 1:
        raise ValueError("mixed frozen feature schemas; build separate datasets")
    accepted = [row for rows in partitions.values() for row in rows]
    accepted_ids = {row["record_id"] for row in accepted}
    accepted_raw = [record for record in records if record.get("record_id") in accepted_ids]
    missing_costs = Counter()
    for record in records:
        for arm_name, value in record.get("arms", {}).items():
            for cost_name in ("model_calls", "physical_attempts", "input_tokens", "output_tokens", "api_cost", "tool_calls", "tool_cost", "wall_seconds"):
                if value.get("costs", {}).get(cost_name) is None:
                    missing_costs[arm_name + ":" + cost_name] += 1
    semantic_composition = {}
    for split, rows in partitions.items():
        trial_ids = {row["trial_id"] for row in rows}
        semantic_composition[split] = {axis: dict(Counter(
            manifest.get("semantic_metadata", {}).get(trial_id, {}).get(axis, "unknown")
            for trial_id in trial_ids)) for axis in ("family", "mechanism", "template")}
    summary = {"schema_version": "dataset-quality.v2", "kind": kind, "planned": len(records),
               "completed": len(accepted), "excluded": len(rejected),
               "planned_trials": len({record.get("trial", {}).get("trial_id") for record in records}),
               "completed_trials": len({row["trial_id"] for row in accepted}),
               "exclusion_reasons": dict(Counter(r["reason"] for r in rejected)),
               "missing_costs_by_arm": dict(missing_costs),
               "split_rows": {k: len(v) for k, v in partitions.items()},
               "split_groups": {k: len({r["group_id"] for r in v}) for k, v in partitions.items()},
               "task_count": len({r["task_id"] for r in accepted}),
               "source_task_count": len({record["trial"]["memory"]["source_task_id"] for record in accepted_raw}),
               "source_trace_count": len({record["trial"]["memory"]["source_trace_id"] for record in accepted_raw}),
               "source_disjoint": manifest["source_disjoint"],
               "frozen_before_collection": manifest["frozen_before_collection"],
               "rho_e_coverage": dict(Counter(str((r["checks"]["rho"], r["checks"]["e"])) for r in accepted)),
               "sampling_weight_range": ([min(r["sampling_weight"] for r in accepted),
                                            max(r["sampling_weight"] for r in accepted)] if accepted else None),
               "semantic_composition": semantic_composition,
               "sampling_weight_contract": "caller-declared natural/sampled density ratio; no frequency inferred from outcomes",
               "manifest_hash": manifest["manifest_hash"]}
    result = {"partitions": partitions, "rejected": rejected, "manifest": manifest, "summary": summary}
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("refusing to overwrite a dataset; select an empty output directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in partitions.items():
            write_jsonl(output_dir / (split + ".jsonl"), rows)
        write_jsonl(output_dir / "rejected.jsonl", rejected)
        write_json(output_dir / "split-manifest.json", manifest)
        write_json(output_dir / "quality.json", summary)
        write_json(output_dir / "checksums.json", {path.name: digest(path.read_text(encoding="utf-8"))
                                                   for path in output_dir.iterdir() if path.is_file()})
    return result


def load_collection(path: str | Path) -> list[dict]:
    """Read collector JSON records or a JSONL file, excluding prepared caches."""
    path = Path(path)
    if path.is_file():
        return read_jsonl(path) if path.suffix == ".jsonl" else [read_json(path)]
    rows = []
    for filename in sorted(path.glob("*.json")):
        value = read_json(filename)
        if "schema_version" in value and "record_id" in value:
            rows.append(value)
    return rows
