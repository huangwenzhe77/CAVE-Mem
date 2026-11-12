"""Frozen-policy evaluation with separate paired and logged estimators.

Intervals resample independent provenance groups; checkpoint replicates are never
treated as independent tasks. These are empirical diagnostics, not formal safety
guarantees. A learned policy must be evaluated on a genuinely untouched cohort.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from .contracts import FeatureRecord, StudySpec, digest, utility
from .learning import (_disjoint, _validate_crossfit, _validate_rows,
                       dr_pseudo_outcomes, row_weights)


def cluster_bootstrap(values, groups, *, weights=None, samples=1000,
                      confidence=.95, seed=42):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and aligned with groups")
    if type(samples) is not int or samples < 2 or not 0 < confidence < 1:
        raise ValueError("bootstrap needs samples >= 2 and confidence in (0,1)")
    groups = [str(g) for g in groups]
    independent = sorted(set(groups))
    if len(independent) < 2:
        raise ValueError("cluster uncertainty requires at least two independent groups")
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != values.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("invalid bootstrap weights")
    w = w / w.sum()
    # Preaggregate numerator/denominator: equivalent to retaining every row of
    # a sampled group, and much cheaper than materializing duplicated records.
    numerators = np.asarray([np.sum(w[np.asarray(groups) == g] * values[np.asarray(groups) == g]) for g in independent])
    denominators = np.asarray([np.sum(w[np.asarray(groups) == g]) for g in independent])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(independent), size=(samples, len(independent)))
    draws = numerators[draws].sum(axis=1) / denominators[draws].sum(axis=1)
    tail = (1 - confidence) / 2
    return {"estimate": float(np.sum(w * values)),
            "lower": float(np.quantile(draws, tail)), "upper": float(np.quantile(draws, 1 - tail)),
            "confidence": confidence, "independent_groups": len(independent),
            "bootstrap_samples": samples, "method": "percentile_cluster_bootstrap"}


def policy_probabilities(policy, rows):
    if hasattr(policy, "predict_probability"):
        p = policy.predict_probability(rows)
    elif callable(policy):
        p = [policy(row) for row in rows]
    else:
        raise TypeError("policy must expose predict_probability(rows) or be callable")
    p = np.asarray(p, dtype=float)
    if p.shape != (len(rows),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("policy probabilities must be finite and in [0,1]")
    return p


def _freeze_check(policy, rows, allow_selection=False):
    for attr in ("training_rows",):
        if hasattr(policy, attr):
            _disjoint(getattr(policy, attr), rows, label="policy fitting/evaluation")
    model = getattr(policy, "outcome_model", None)
    if model is not None:
        _disjoint(getattr(model, "training_rows", []), rows, label="outcome fitting/evaluation")
        _disjoint(getattr(model, "calibration_rows", []), rows, label="calibration/evaluation")
    if not allow_selection:
        _disjoint(getattr(policy, "selection_rows", []), rows, label="policy selection/evaluation")
        if any(r.get("split", "audit") not in ("audit", "test") for r in rows):
            raise ValueError("frozen-policy evaluation needs untouched audit or test rows")
        selection_hash = getattr(policy, "report", {}).get("constraint_data_hash")
        if selection_hash is not None and selection_hash == digest(rows):
            raise ValueError("constraint selection cohort is not an untouched evaluation cohort")


def _metrics(series, rows, *, bootstrap_samples, confidence, seed):
    weights = row_weights(rows)
    groups = [r["group_id"] for r in rows]
    return {name: cluster_bootstrap(values, groups, weights=weights,
                                   samples=bootstrap_samples, confidence=confidence, seed=seed + i)
            for i, (name, values) in enumerate(series.items())}


def evaluate_paired(policy, rows, *, bootstrap_samples=1000, confidence=.95,
                    seed=42, allow_selection=False):
    """Exact expected mixture over the two observed forks, not sampled actions.

    regression = E[pi(1|x) H]; rescue is likewise policy weighted. extra_compute
    is Eq. 11's E[c0 + pi delta_cost], not absolute full-episode expenditure.
    """
    _validate_rows(rows, "paired")
    _freeze_check(policy, rows, allow_selection)
    p, weights = policy_probabilities(policy, rows), row_weights(rows)
    r0, r1 = (np.asarray([r[key] for r in rows]) for key in ("r0", "r1"))
    y0, y1 = (np.asarray([r[key] for r in rows]) for key in ("y0", "y1"))
    series = {"value": (1 - p) * r0 + p * r1,
              "silent_value": r0, "improvement": p * (r1 - r0),
              "success": (1 - p) * y0 + p * y1,
              "regression": p * np.asarray([r["H"] for r in rows]),
              "rescue": p * ((y0 == 0) & (y1 == 1)),
              "extra_compute": np.asarray([r["c0"] for r in rows]) + p * np.asarray([r["delta_cost"] for r in rows]),
              "reminder_rate": p}
    if all("cost0" in r and "cost1" in r and r["cost0"] is not None and r["cost1"] is not None for r in rows):
        series["absolute_compute"] = (1 - p) * np.asarray([r["cost0"] for r in rows]) + p * np.asarray([r["cost1"] for r in rows])
    # e=0 stratum is operational exposure, not a promise of terminal success.
    safe = np.asarray([r.get("checks", {}).get("v") == 1 and
                       r.get("checks", {}).get("rho") == 1 and
                       r.get("checks", {}).get("e") == 0 for r in rows])
    report = {"estimator": "paired_fork_policy_mixture", "rows_hash": digest(rows),
              "rows": len(rows), "task_count": len({r["task_id"] for r in rows}),
              "independent_groups": len({r["group_id"] for r in rows}),
              "checkpoints": len({(r["task_id"], r["checkpoint_id"]) for r in rows}),
              "metrics": _metrics(series, rows, bootstrap_samples=bootstrap_samples, confidence=confidence, seed=seed),
              "unnecessary_reminder_rate": float(np.sum(weights[safe] * p[safe]) / np.sum(weights[safe])) if safe.any() else None,
              "coupling_ids": sorted({str(r.get("coupling", "declared_by_dataset")) for r in rows}),
              "evaluation_role": "selection" if allow_selection else "untouched",
              "limitations": ["paired regression depends on the declared fork coupling",
                              "cluster bootstrap is an empirical diagnostic, not a trajectory-level guarantee"]}
    return report


def effective_sample_size(weights):
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("ESS weights must be nonnegative and finite")
    denominator = np.sum(values ** 2)
    return float(np.sum(values) ** 2 / denominator) if denominator else 0.


def evaluate_logged(policy, rows, crossfit, *, bootstrap_samples=1000, confidence=.95,
                    seed=42, clipping_grid=(None, 10., 20.)):
    """DR value of a frozen policy with OOF nuisance predictions, Eq. 8.

    No H, paired rescue, or individual causal harm is constructed from these logs.
    Clipping sensitivity is reported beside the unclipped estimator, never hidden.
    """
    _validate_rows(rows, "single")
    _validate_crossfit(rows, crossfit)
    _freeze_check(policy, rows)
    p = policy_probabilities(policy, rows)
    observed_p = np.asarray([p[i] if r["z"] == 1 else 1 - p[i] for i, r in enumerate(rows)])
    propensity = np.asarray([r["b"] for r in rows])
    ipw = observed_p / propensity
    natural_ipw = row_weights(rows) * ipw
    group_mass = [float(natural_ipw[np.asarray([r["group_id"] == group for r in rows])].sum())
                  for group in sorted({r["group_id"] for r in rows})]
    diagnostics = {"ess": effective_sample_size(ipw), "max_importance_weight": float(ipw.max()),
                   "mean_importance_weight": float(np.mean(ipw)), "min_observed_propensity": float(propensity.min()),
                   "arm_counts": {str(z): sum(r["z"] == z for r in rows) for z in (0, 1)},
                   "arm_ess": {str(z): effective_sample_size(ipw * np.asarray([r["z"] == z for r in rows])) for z in (0, 1)},
                   "natural_weighted_row_ess": effective_sample_size(natural_ipw),
                   "natural_weighted_group_mass_ess": effective_sample_size(group_mass),
                   "independent_groups": len(group_mass),
                   "ess_note": "row ESS is an importance-weight diagnostic, not an independent-task count"}
    reports = []
    if None not in clipping_grid:
        raise ValueError("clipping sensitivity must include the unclipped estimator")
    for clip in clipping_grid:
        pseudo = dr_pseudo_outcomes(rows, crossfit["mu"], clipping=clip)
        silent = pseudo[:, 0]
        series = {"value": (1 - p) * pseudo[:, 0] + p * pseudo[:, 1],
                  "silent_value": silent, "improvement": p * (pseudo[:, 1] - pseudo[:, 0]),
                  "reminder_rate": p}
        reports.append({"inverse_propensity_clip": clip,
                        "bias_note": "unclipped" if clip is None else "clipped residual correction may be biased",
                        "metrics": _metrics(series, rows, bootstrap_samples=bootstrap_samples, confidence=confidence, seed=seed)})
    return {"estimator": "logged_single_arm_DR", "rows_hash": digest(rows),
            "crossfit_hash": digest(crossfit), "rows": len(rows), "diagnostics": diagnostics,
            "sensitivity": reports, "paired_harm_identified": False,
            "regression": None, "rescue": None,
            "limitations": ["known randomized propensities and overlap are required",
                            "nuisance cross-fitting does not allow evaluating a policy on its own training cohort",
                            "binary single-arm outcomes do not identify the joint paired regression event"]}


def calibration_report(model, rows, *, bins=10):
    _validate_rows(rows, "paired")
    _disjoint(model.training_rows, rows, label="outcome fitting/calibration evaluation")
    _disjoint(getattr(model, "calibration_rows", []), rows, label="calibration fitting/evaluation")
    if type(bins) is not int or bins < 2:
        raise ValueError("at least two calibration bins are required")
    predictions = model.predict(rows)
    weights = row_weights(rows)
    y, d = np.asarray([r["H"] for r in rows]), np.asarray([r["D"] for r in rows])
    risk = predictions["risk"]
    assignments = np.minimum((risk * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.
    for index in range(bins):
        mask = assignments == index
        if not mask.any():
            continue
        mass = weights[mask].sum()
        predicted, observed = np.sum(weights[mask] * risk[mask]) / mass, np.sum(weights[mask] * y[mask]) / mass
        reliability.append({"bin": index, "rows": int(mask.sum()), "weight": float(mass),
                            "predicted": float(predicted), "observed": float(observed)})
        ece += mass * abs(predicted - observed)
    return {"advantage_mse": float(np.sum(weights * (predictions["tau"] - d) ** 2)),
            "risk_brier": float(np.sum(weights * (risk - y) ** 2)), "risk_ece": float(ece),
            "reliability": reliability,
            "positive_lower_coverage": float(np.sum(weights * (predictions["tau_lower"] > 0))),
            "note": "bin calibration and residual bounds do not imply conditional safety"}


def compare_paired(policy_a, policy_b, rows, **bootstrap_options):
    """Task/group-clustered intervals for policy differences on the same forks."""
    _validate_rows(rows, "paired")
    _freeze_check(policy_a, rows)
    _freeze_check(policy_b, rows)
    delta = policy_probabilities(policy_a, rows) - policy_probabilities(policy_b, rows)
    metrics = {"improvement_difference": delta * np.asarray([r["D"] for r in rows]),
               "regression_difference": delta * np.asarray([r["H"] for r in rows]),
               "compute_difference": delta * np.asarray([r["delta_cost"] for r in rows])}
    return {key: cluster_bootstrap(value, [r["group_id"] for r in rows],
                                   weights=row_weights(rows), **bootstrap_options) for key, value in metrics.items()}


def evaluate_deployment(records, policy, *, bootstrap_samples=1000, confidence=.95, seed=42,
                        source_disjoint=True, allow_selection=False):
    """Actual frozen-policy single-intervention rollouts; no paired H is invented.

    The policy is required to check training/calibration/selection lineage and
    compare recorded intervention probabilities with the supplied frozen policy.
    Failed infrastructure and unknown utility/cost are explicit coverage limits.
    """
    if not records:
        raise ValueError("deployment evaluation requires actual rollout records")
    if type(source_disjoint) is not bool or type(allow_selection) is not bool:
        raise ValueError("declare boolean source-disjoint and selection flags")
    rows, arms, probabilities, excluded, seen = [], [], [], [], set()
    for record in records:
        if record.get("schema_version") != "deployment-intervention.v2":
            raise ValueError("deployment evaluation only accepts deployment rollout records")
        if record.get("record_hash") != digest({k: v for k, v in record.items() if k != "record_hash"}):
            raise ValueError("deployment record checksum mismatch")
        if not record.get("valid"):
            excluded.append({"record_id": record.get("record_id"), "error": record.get("error"),
                             "reason": "invalid_deployment_rollout"})
            continue
        trial, decision = record["trial"], record["decision"]
        z = decision["z"]
        if type(z) is not int or z not in (0, 1):
            raise ValueError("deployment decision must have binary z")
        p = float(decision["probability"])
        if not np.isfinite(p) or not 0 <= p <= 1 or (p == 0 and z == 1) or (p == 1 and z == 0):
            raise ValueError("realized decision is incompatible with the deployed policy")
        if set(record["arms"]) != {"remind" if z else "silent"}:
            raise ValueError("deployment must contain only the actual executed arm")
        arm = record["arms"]["remind" if z else "silent"]
        if arm["status"] not in ("terminal", "invalid_action", "budget_exhausted") or type(arm["success"]) is not bool:
            raise ValueError("deployment arm needs a valid fixed-budget binary outcome")
        if arm["status"] != "terminal" and arm["success"]:
            raise ValueError("nonterminal deployment arm cannot claim success")
        features = FeatureRecord(**trial["features"])
        identity = (trial["trial_id"], record.get("replicate", 0))
        if identity in seen:
            raise ValueError("duplicate deployment trial/replicate")
        seen.add(identity)
        checks = trial["checks"]
        dependencies = list(trial["group_keys"])
        if not source_disjoint:
            source_task = (trial.get("memory") or {}).get("source_task_id")
            dependencies = [key for key in dependencies if not key.startswith(("source_task:", "source_trace:", "memory:"))
                            and key != "task:" + str(source_task)]
        row = {"group_id": record.get("evaluation_group_id", "pending"), "group_keys": dependencies,
               "task_id": trial["task_id"], "checkpoint_id": trial["checkpoint_id"],
               "trial_id": trial["trial_id"], "x": list(features.values),
               "replicate": record.get("replicate", 0),
               "feature_names": list(features.names), "feature_version": features.version,
               "sampling_weight": trial.get("sampling_weight", 1.),
               "source_disjoint": source_disjoint,
               "split": record.get("evaluation_split", record.get("population", "audit")),
               "eligible": trial.get("memory") is not None and checks.get("v") == 1, "checks": checks}
        accepted_splits = ("selection", "validation", "audit", "test") if allow_selection else ("audit", "test")
        if row["split"] not in accepted_splits:
            raise ValueError("deployment summary needs a declared untouched audit/test population")
        recomputed = utility(arm["success"], arm["costs"], StudySpec(**record["protocol"]))
        if (recomputed is None) != (arm.get("utility") is None) or (
                recomputed is not None and not np.isclose(recomputed, arm["utility"])):
            raise ValueError("deployment utility disagrees with the declared costs/reward")
        rows.append(row)
        arms.append(arm)
        probabilities.append(p)
    if not rows:
        return {"estimator": "actual_deployment_rollouts", "planned": len(records), "completed": 0,
                "excluded": excluded, "metrics": {}, "paired_harm_identified": False}
    # Reconstruct connected provenance components for cluster resampling. Training
    # lineage retains the actual split dependencies, so shared-bank designs can
    # omit source dependencies explicitly in their training manifests.
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        dependencies = set(row["group_keys"]) | {"task:" + row["task_id"], "checkpoint:" + row["checkpoint_id"]}
        for key in dependencies:
            if key in owners:
                a, b = root(i), root(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = i
    components = {}
    for i in range(len(rows)):
        components.setdefault(root(i), []).append(i)
    for indices in components.values():
        provided_ids = {rows[i]["group_id"] for i in indices if rows[i]["group_id"] != "pending"}
        if len(provided_ids) > 1:
            raise ValueError("deployment annotations split one connected provenance component")
        dependencies = sorted({key for i in indices for key in rows[i]["group_keys"]} |
                              {"task:" + rows[i]["task_id"] for i in indices} |
                              {"checkpoint:" + rows[i]["checkpoint_id"] for i in indices})
        group_id = next(iter(provided_ids)) if provided_ids else "deployment-group-" + digest(dependencies)[:24]
        for i in indices:
            rows[i]["group_id"] = group_id
    _freeze_check(policy, rows, allow_selection=allow_selection)
    eligible_indices = [i for i, row in enumerate(rows) if row["eligible"]]
    if eligible_indices:
        expected = policy_probabilities(policy, [rows[i] for i in eligible_indices])
        if not np.allclose(expected, np.asarray(probabilities)[eligible_indices], rtol=1e-6, atol=1e-8):
            raise ValueError("deployment probabilities differ from the supplied frozen policy")
    if any(probabilities[i] != 0 for i, row in enumerate(rows) if not row["eligible"]):
        raise ValueError("ineligible deployment decisions must remain silent")
    series = {"success": np.asarray([float(arm["success"]) for arm in arms]),
              "intervention_rate": np.asarray([float(r["decision"]["z"]) for r in records if r.get("valid")]),
              "expected_intervention_rate": np.asarray(probabilities)}
    known_utility = [arm.get("utility") is not None for arm in arms]
    if all(known_utility):
        series["value"] = np.asarray([arm["utility"] for arm in arms])
    cost_keys = sorted({key for arm in arms for key in arm["costs"]})
    coverage = {}
    for key in cost_keys:
        measured = [arm["costs"].get(key) for arm in arms]
        coverage[key] = sum(v is not None for v in measured) / len(arms)
        if all(v is not None for v in measured):
            if not np.isfinite(measured).all() or np.any(np.asarray(measured) < 0):
                raise ValueError("invalid deployment costs")
            series["cost_" + key] = np.asarray(measured)
    return {"estimator": "actual_deployment_rollouts", "records_hash": digest(records),
            "planned": len(records), "completed": len(rows), "excluded": excluded,
            "metrics": _metrics(series, rows, bootstrap_samples=bootstrap_samples, confidence=confidence, seed=seed),
            "utility_coverage": sum(known_utility) / len(rows), "cost_coverage": coverage,
            "source_disjoint": source_disjoint,
            "evaluation_role": "selection" if allow_selection else "untouched",
            "paired_harm_identified": False, "regression": None, "rescue": None,
            "lineage_checked": True,
            "limitations": ["actual single-arm rollouts do not reveal individual rescue/regression",
                            "no value or cost estimate is imputed for a metric with missing required usage",
                            "infrastructure failures are excluded with explicit coverage, not converted to task failure"]}
