"""CPU outcome models and offline binary policies for proposal-2, Eqs. 7--14.

NumPy is an optional training dependency. Inputs are already frozen feature vectors;
this module never embeds text, reads environment state, or updates the acting LLM.
Paired and single-arm observations deliberately have separate entry points.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as exc:  # Only the optional learning module needs NumPy.
    raise ImportError("Install cave-mem-v2[train] to train CPU models") from exc

from .contracts import digest, plain, read_json, write_json

EPS = 1e-7


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_rows(rows, kind=None, *, required_split=None):
    if not rows:
        raise ValueError("a nonempty independent data partition is required")
    names = tuple(rows[0]["feature_names"])
    version = rows[0].get("feature_version", "decision-features.v1")
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be nonempty and unique")
    seen = set()
    for row in rows:
        if type(row.get("source_disjoint", True)) is not bool:
            raise ValueError("source-disjoint experiment scope must be boolean")
        if row.get("source_disjoint", True) != rows[0].get("source_disjoint", True):
            raise ValueError("mixed source-disjoint experiment scopes")
        if tuple(row["feature_names"]) != names or len(row["x"]) != len(names):
            raise ValueError("inconsistent frozen feature schema")
        if row.get("feature_version", "decision-features.v1") != version:
            raise ValueError("mixed feature versions in one cohort")
        if not all(np.isfinite(_number(v, "feature")) for v in row["x"]):
            raise ValueError("nonfinite feature")
        for key in ("group_id", "task_id", "checkpoint_id", "trial_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing {key}")
        identity = (row["trial_id"], row.get("replicate", 0))
        if identity in seen:
            raise ValueError("duplicate trial/replicate")
        seen.add(identity)
        accepted = ("selection", "validation") if required_split == "selection" else (required_split,)
        if required_split is not None and row.get("split", required_split) not in accepted:
            raise ValueError(f"this operation requires {required_split} rows")
        weight = _number(row.get("sampling_weight", 1.0), "sampling_weight")
        if weight <= 0:
            raise ValueError("sampling weights must be positive")
        if kind == "paired":
            for key in ("D", "H", "c0", "delta_cost", "y0", "y1", "r0", "r1"):
                _number(row.get(key), key)
            if row["H"] not in (0, 1) or row["y0"] not in (0, 1) or row["y1"] not in (0, 1):
                raise ValueError("paired success/regression labels must be binary")
            if row["H"] != int(row["y0"] == 1 and row["y1"] == 0):
                raise ValueError("H must be paired success-to-failure regression")
            if not np.isclose(row["D"], row["r1"] - row["r0"]):
                raise ValueError("D must be the difference in terminal utilities")
            if row["c0"] < 0:
                raise ValueError("overhead c0 cannot be negative")
        if kind == "single":
            if "H" in row or any(k in row for k in ("y0", "y1", "D")):
                raise ValueError("single-arm data cannot claim observed paired outcomes")
            if type(row.get("z")) is not int or row["z"] not in (0, 1):
                raise ValueError("single-arm z must be integer 0 or 1")
            if not 0 < _number(row.get("b"), "observed treatment propensity") < 1:
                raise ValueError("randomized treatment requires overlap 0 < b < 1")
            _number(row.get("r"), "reward")
    return names


def _disjoint(left, right, *, label="partitions"):
    scopes = {r.get("source_disjoint", True) for r in [*left, *right]}
    if len(scopes) > 1:
        raise ValueError(f"{label} use different source-disjoint experiment scopes")
    for key in ("group_id", "task_id", "checkpoint_id", "trial_id"):
        if {r[key] for r in left} & {r[key] for r in right}:
            raise ValueError(f"{label} overlap by {key}")
    left_dependencies = {key for row in left for key in row.get("group_keys", ())}
    right_dependencies = {key for row in right for key in row.get("group_keys", ())}
    if left_dependencies & right_dependencies:
        raise ValueError(f"{label} overlap by declared provenance dependency")


def _lineage(rows):
    return [{key: copy.deepcopy(row[key]) for key in
             ("group_id", "task_id", "checkpoint_id", "trial_id", "group_keys", "source_disjoint") if key in row}
            for row in rows]


def row_weights(rows):
    """Inverse sampling weights with fork replicates averaged per checkpoint.

    Multiple checkpoints remain separate sampled decisions; inference clusters on
    task/group. sampling_weight is the declared natural/sampled density ratio.
    """
    counts = Counter((r["task_id"], r["checkpoint_id"]) for r in rows)
    weights = np.asarray([float(r.get("sampling_weight", 1)) /
                          counts[(r["task_id"], r["checkpoint_id"])] for r in rows])
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid sampling weights")
    return weights / weights.sum()


def _schema(rows):
    return {"names": list(_validate_rows(rows)),
            "version": rows[0].get("feature_version", "decision-features.v1")}


def _matrix(rows, schema):
    _validate_rows(rows)
    if list(rows[0]["feature_names"]) != schema["names"]:
        raise ValueError("model feature names/order changed")
    if any(r.get("feature_version", "decision-features.v1") != schema["version"] for r in rows):
        raise ValueError("model feature version changed")
    return np.asarray([r["x"] for r in rows], dtype=float)


class _MLP:
    """A tanh hidden layer, independently initialized heads, full-batch Adam."""
    def __init__(self, inputs, hidden, outputs, seed=42):
        if min(inputs, hidden, outputs) < 1:
            raise ValueError("network sizes must be positive")
        rng = np.random.default_rng(seed)
        self.params = {"w1": rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden)),
                       "b1": np.zeros(hidden),
                       "w2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, outputs)),
                       "b2": np.zeros(outputs)}

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden @ self.params["w2"] + self.params["b2"], hidden

    def backward(self, x, hidden, gradient, l2=0):
        gh = (gradient @ self.params["w2"].T) * (1 - hidden ** 2)
        return {"w1": x.T @ gh + l2 * self.params["w1"], "b1": gh.sum(axis=0),
                "w2": hidden.T @ gradient + l2 * self.params["w2"], "b2": gradient.sum(axis=0)}

    def state(self):
        return {key: value.tolist() for key, value in self.params.items()}

    @classmethod
    def restore(cls, state):
        arrays = {k: np.asarray(v, dtype=float) for k, v in state.items()}
        obj = cls(arrays["w1"].shape[0], arrays["w1"].shape[1], arrays["w2"].shape[1])
        for key in obj.params:
            if arrays[key].shape != obj.params[key].shape or not np.isfinite(arrays[key]).all():
                raise ValueError("invalid network checkpoint")
        obj.params = arrays
        return obj


class _Adam:
    def __init__(self, params, rate):
        if not 0 < rate <= 1:
            raise ValueError("learning rate must be in (0,1]")
        self.rate, self.step = rate, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update(self, params, gradients):
        self.step += 1
        for key, gradient in gradients.items():
            if not np.isfinite(gradient).all():
                raise ValueError("nonfinite training gradient")
            gradient = np.clip(gradient, -10, 10)
            self.m[key] = .9 * self.m[key] + .1 * gradient
            self.v[key] = .999 * self.v[key] + .001 * gradient ** 2
            params[key] -= self.rate * (self.m[key] / (1 - .9 ** self.step)) / (
                np.sqrt(self.v[key] / (1 - .999 ** self.step)) + 1e-8)


def _training_rows(rows):
    _validate_rows(rows, required_split="train")
    if len({r["group_id"] for r in rows}) < 2:
        raise ValueError("training needs at least two independent groups")


def _fit_network(network, x, targets, weights, *, epochs, learning_rate, risk_column=None, l2=1e-4,
                 head_weights=None):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be positive")
    optimizer = _Adam(network.params, learning_rate)
    for _ in range(epochs):
        out, hidden = network.forward(x)
        gradient = 2 * (out - targets)
        if risk_column is not None:
            gradient[:, risk_column] = sigmoid(out[:, risk_column]) - targets[:, risk_column]
        if head_weights is not None:
            gradient *= np.asarray(head_weights)[None, :]
        gradient *= weights[:, None]
        optimizer.update(network.params, network.backward(x, hidden, gradient, l2))


def _checkpoint(kind, payload, path):
    record = {"artifact_version": "learning.v1", "kind": kind, **payload}
    record["checksum"] = digest(record)
    write_json(path, record)


def _load(path, kind):
    record = read_json(path)
    checksum = record.pop("checksum", None)
    if checksum != digest(record) or record.get("kind") != kind or record.get("artifact_version") != "learning.v1":
        raise ValueError("model checkpoint checksum/type/version mismatch")
    return record


class OutcomeModel:
    """Bootstrap ensemble of same-size MLPs for tau, regression, delta cost, c0.

    tau targets D already include the declared utility cost penalty. The separate
    cost head is used for Eq. 11, never subtracted from D a second time.
    """
    def __init__(self, hidden=16, ensemble_size=5, seed=42):
        if type(ensemble_size) is not int or ensemble_size < 2:
            raise ValueError("at least two ensemble members are required")
        self.hidden, self.ensemble_size, self.seed = hidden, ensemble_size, seed
        self.networks = []
        self.calibration = None

    def fit(self, rows, *, epochs=300, learning_rate=.01, risk_loss_weight=1., cost_loss_weight=1.):
        _training_rows(rows)
        _validate_rows(rows, "paired")
        if _number(risk_loss_weight, "risk loss weight") < 0 or _number(cost_loss_weight, "cost loss weight") < 0:
            raise ValueError("loss weights cannot be negative")
        self.schema = _schema(rows)
        raw = _matrix(rows, self.schema)
        self.mean, self.scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
        x = (raw - self.mean) / self.scale
        targets = np.asarray([[r[k] for k in ("D", "H", "delta_cost", "c0")] for r in rows])
        self.networks, self.calibration = [], None
        self.training_rows = _lineage(rows)
        self.training_hash = digest(rows)
        self.fit_config = {"epochs": epochs, "learning_rate": learning_rate,
                           "risk_loss_weight": risk_loss_weight, "cost_loss_weight": cost_loss_weight}
        self.training_groups = sorted({r["group_id"] for r in rows})
        weights = row_weights(rows)
        # Cost units can be dollars, calls, or thousands of tokens. Standardize
        # only cost targets so changing their unit does not swamp Eq. 7's heads.
        self.target_offset, self.target_scale = np.zeros(4), np.ones(4)
        for column in (2, 3):
            self.target_offset[column] = np.sum(weights * targets[:, column])
            self.target_scale[column] = max(1., float(np.sqrt(np.sum(weights * (targets[:, column] - self.target_offset[column]) ** 2))))
        targets = (targets - self.target_offset) / self.target_scale
        groups = sorted({r["group_id"] for r in rows})
        rng = np.random.default_rng(self.seed)
        for index in range(self.ensemble_size):
            counts = Counter(rng.choice(groups, len(groups), replace=True))
            boot = weights * np.asarray([counts.get(r["group_id"], 0) for r in rows])
            boot /= boot.sum()
            net = _MLP(x.shape[1], self.hidden, 4, self.seed + index)
            _fit_network(net, x, targets, boot, epochs=epochs, learning_rate=learning_rate, risk_column=1,
                         head_weights=(1., risk_loss_weight, cost_loss_weight, cost_loss_weight))
            self.networks.append(net)
        return self

    def predict(self, rows):
        if not self.networks:
            raise ValueError("outcome model has not been fitted")
        x = (_matrix(rows, self.schema) - self.mean) / self.scale
        raw = np.stack([net.forward(x)[0] for net in self.networks])
        for column in (2, 3):
            raw[:, :, column] = raw[:, :, column] * self.target_scale[column] + self.target_offset[column]
        tau = raw[:, :, 0].mean(axis=0)
        risk_logits = raw[:, :, 1].mean(axis=0)
        risk = sigmoid(risk_logits)
        lower = np.full(len(rows), -np.inf)
        if self.calibration is not None:
            risk = sigmoid(self.calibration["slope"] * risk_logits + self.calibration["intercept"])
            lower = tau - self.calibration["advantage_residual_quantile"]
        return {"tau": tau, "risk": risk, "delta_cost": raw[:, :, 2].mean(axis=0),
                "c0": np.maximum(0, raw[:, :, 3].mean(axis=0)),
                "tau_ensemble_sd": raw[:, :, 0].std(axis=0, ddof=1),
                "tau_lower": lower, "risk_logit": risk_logits}

    def calibrate(self, rows, *, alpha=.1, epochs=400, learning_rate=.03):
        _validate_rows(rows, "paired", required_split="calibration")
        _disjoint(self.training_rows, rows, label="training/calibration")
        if len({r["group_id"] for r in rows}) < 2 or not 0 < alpha < 1:
            raise ValueError("calibration requires independent groups and 0 < alpha < 1")
        predictions = self.predict(rows)
        w, y = row_weights(rows), np.asarray([r["H"] for r in rows])
        a, b = 1., 0.
        logits = predictions["risk_logit"]
        for _ in range(epochs):
            error = sigmoid(a * logits + b) - y
            a = max(0., a - learning_rate * np.sum(w * error * logits))
            b -= learning_rate * np.sum(w * error)
        residuals = {}
        for row, prediction in zip(rows, predictions["tau"]):
            key = row["group_id"]
            residuals[key] = max(residuals.get(key, 0.), abs(float(row["D"] - prediction)))
        # Group-max residual diagnostic: no claimed conditional/trajectory guarantee.
        quantile = float(np.quantile(list(residuals.values()), min(1., np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals)), method="higher"))
        self.calibration = {"slope": a, "intercept": b, "alpha": alpha,
                            "epochs": epochs, "learning_rate": learning_rate,
                            "advantage_residual_quantile": quantile,
                            "calibration_hash": digest(rows), "groups": len(residuals),
                            "claim": "held-out group-max residual diagnostic, not conditional coverage"}
        self.calibration_rows = _lineage(rows)
        calibrated = self.predict(rows)["risk"]
        return {**self.calibration, "risk_brier": float(np.sum(w * (calibrated - y) ** 2)),
                "observed_regression": float(np.sum(w * y)),
                "predicted_regression": float(np.sum(w * calibrated))}

    def save(self, path):
        if not self.networks:
            raise ValueError("cannot save an unfitted model")
        _checkpoint("outcome", {"hidden": self.hidden, "ensemble_size": self.ensemble_size, "seed": self.seed,
                    "schema": self.schema, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                    "target_offset": self.target_offset.tolist(), "target_scale": self.target_scale.tolist(),
                    "networks": [n.state() for n in self.networks], "training_hash": self.training_hash,
                    "training_rows": self.training_rows, "training_groups": self.training_groups,
                    "fit_config": self.fit_config,
                    "calibration": self.calibration, "calibration_rows": getattr(self, "calibration_rows", [])}, path)

    @classmethod
    def load(cls, path):
        value = _load(path, "outcome")
        obj = cls(value["hidden"], value["ensemble_size"], value["seed"])
        for key in ("schema", "training_hash", "training_rows", "training_groups", "calibration", "calibration_rows"):
            setattr(obj, key, value[key])
        obj.mean, obj.scale = np.asarray(value["mean"]), np.asarray(value["scale"])
        obj.fit_config = value.get("fit_config", {})
        obj.target_offset, obj.target_scale = np.asarray(value.get("target_offset", [0.] * 4)), np.asarray(value.get("target_scale", [1.] * 4))
        obj.networks = [_MLP.restore(n) for n in value["networks"]]
        return obj


def eligible(row):
    """Only verified source evidence is a hard eligibility rule in the main gate."""
    if "eligible" in row:
        return row["eligible"] is True
    values = dict(zip(row["feature_names"], row["x"]))
    # Data adapters must declare eligibility rather than infer it from outcomes.
    if "validity" not in values and "v" not in values:
        raise ValueError("decision row must declare eligible or a validity feature")
    return values.get("validity", values.get("v")) == 1 and values.get("v_unknown", 0) == 0


class BinaryPolicy:
    """Same hidden capacity for supervised warm start and direct-policy learning."""
    def __init__(self, hidden=16, seed=42):
        self.hidden, self.seed = hidden, seed
        self.network = None
        self.outcome_model = None
        self.safety = {"require_positive_lower": False, "risk_threshold": 1.0}

    def warm_start(self, rows, outcome_model, *, epochs=300, learning_rate=.01,
                   risk_threshold=1.0, require_positive_lower=True):
        _training_rows(rows)
        if not 0 <= _number(risk_threshold, "optional pointwise risk threshold") <= 1:
            raise ValueError("risk threshold must be in [0,1]")
        if outcome_model.calibration is None:
            raise ValueError("warm start requires a separately calibrated outcome model")
        self.schema = copy.deepcopy(outcome_model.schema)
        self.mean, self.scale = outcome_model.mean.copy(), outcome_model.scale.copy()
        self.outcome_model = copy.deepcopy(outcome_model)
        prediction = outcome_model.predict(rows)
        tau = prediction["tau_lower"] if require_positive_lower else prediction["tau"]
        targets = np.asarray([float(eligible(row)) for row in rows]) * (tau > 0) * (prediction["risk"] <= risk_threshold)
        # Smoothed targets keep the reference policy's support for the KL term.
        targets = .02 + .96 * targets.astype(float)
        x = (_matrix(rows, self.schema) - self.mean) / self.scale
        self.network = _MLP(x.shape[1], self.hidden, 1, self.seed)
        _fit_network(self.network, x, targets[:, None], row_weights(rows), epochs=epochs,
                     learning_rate=learning_rate, risk_column=0)
        self.reference_state = self.network.state()
        self.training_hash, self.training_rows = digest(rows), _lineage(rows)
        self.safety = {"require_positive_lower": require_positive_lower, "risk_threshold": risk_threshold}
        self.report = {"method": "supervised_warm_start", "same_hidden_capacity": self.hidden,
                       "epochs": epochs, "learning_rate": learning_rate,
                       "positive_targets": int(np.sum(targets > .5))}
        return self

    def warm_start_logged(self, rows, crossfit, *, epochs=300, learning_rate=.01):
        """Single-arm reward-model warm start; does not learn/claim paired risk."""
        _training_rows(rows)
        _validate_rows(rows, "single")
        _validate_crossfit(rows, crossfit)
        self.schema = _schema(rows)
        raw = _matrix(rows, self.schema)
        self.mean, self.scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
        self.outcome_model = None
        mu = np.asarray(crossfit["mu"])
        targets = .02 + .96 * ((mu[:, 1] - mu[:, 0]) > 0).astype(float)
        x = (raw - self.mean) / self.scale
        self.network = _MLP(x.shape[1], self.hidden, 1, self.seed)
        _fit_network(self.network, x, targets[:, None], row_weights(rows), epochs=epochs,
                     learning_rate=learning_rate, risk_column=0)
        self.reference_state = self.network.state()
        self.training_hash = digest(rows)
        self.training_rows = _lineage(rows)
        self.safety = {"require_positive_lower": False, "risk_threshold": 1.}
        self.report = {"method": "single_arm_reward_warm_start", "paired_risk_identified": False,
                       "epochs": epochs, "learning_rate": learning_rate,
                       "warning": "experimental value baseline; no paired-risk or advantage-confidence safeguard"}
        return self

    def _safety_mask(self, rows):
        mask = np.asarray([eligible(row) for row in rows])
        if self.outcome_model is not None:
            pred = self.outcome_model.predict(rows)
            if self.safety["require_positive_lower"]:
                mask &= pred["tau_lower"] > 0
            mask &= pred["risk"] <= self.safety["risk_threshold"]
        return mask

    def predict_probability(self, rows, *, apply_safety=True):
        if self.network is None:
            raise ValueError("policy has not been initialized")
        x = (_matrix(rows, self.schema) - self.mean) / self.scale
        probabilities = sigmoid(self.network.forward(x)[0][:, 0])
        if getattr(self, "force_silent", False):
            return np.zeros(len(rows))
        if apply_safety:
            probabilities *= self._safety_mask(rows)
        return probabilities

    def _optimize(self, rows, advantages, *, constraint_rows=None, epochs=250, learning_rate=.005,
                  kl_weight=.01, risk_budget=.05, compute_budget=2., dual_rate=.1,
                  audit_every=25, bootstrap_samples=200, confidence=.95, constrained=True):
        if self.network is None:
            raise ValueError("initialize the same-capacity supervised policy first")
        if type(epochs) is not int or epochs < 1 or type(audit_every) is not int or audit_every < 1:
            raise ValueError("epochs and audit_every must be positive integers")
        self.force_silent = False
        _training_rows(rows)
        if self.training_hash != digest(rows):
            raise ValueError("policy optimization must use its declared warm-start training rows")
        if constrained:
            if constraint_rows is None or self.outcome_model is None:
                raise ValueError("risk constraints require paired held-out selection data and risk model")
            _validate_rows(constraint_rows, "paired", required_split="selection")
            _disjoint(self.training_rows, constraint_rows, label="training/constraint selection")
            _disjoint(getattr(self.outcome_model, "calibration_rows", []), constraint_rows,
                      label="calibration/constraint selection")
            if len({r["group_id"] for r in constraint_rows}) < 2:
                raise ValueError("constraint selection needs multiple independent groups")
        if not 0 <= risk_budget <= 1 or compute_budget < 0 or kl_weight < 0 or dual_rate < 0:
            raise ValueError("invalid risk/cost/KL/dual configuration")
        x = (_matrix(rows, self.schema) - self.mean) / self.scale
        w, d = row_weights(rows), np.asarray(advantages)
        reference = _MLP.restore(self.reference_state)
        p0 = sigmoid(reference.forward(x)[0][:, 0])
        predictions = self.outcome_model.predict(rows) if constrained else None
        mask = self._safety_mask(rows).astype(float)
        optimizer = _Adam(self.network.params, learning_rate)
        alpha, beta, history, best = 0., 0., [], None
        best_value = -np.inf
        for epoch in range(1, epochs + 1):
            out, hidden = self.network.forward(x)
            raw_p = sigmoid(out[:, 0])
            p = raw_p * mask
            direction = d - kl_weight * (logit(raw_p) - logit(p0))
            if constrained:
                direction -= alpha * predictions["risk"] + beta * predictions["delta_cost"]
            # Objective gradients use smooth expected-risk/cost surrogates. The
            # held-out bootstrap is an audit, never differentiated through.
            gradient = -(w * mask * raw_p * (1 - raw_p) * direction)[:, None]
            optimizer.update(self.network.params, self.network.backward(x, hidden, gradient, l2=1e-5))
            if epoch % audit_every == 0 or epoch == epochs:
                info = {"epoch": epoch, "alpha": alpha, "beta": beta,
                        "training_improvement": float(np.sum(w * p * d))}
                if constrained:
                    from .evaluation import evaluate_paired
                    audit = evaluate_paired(self, constraint_rows, bootstrap_samples=bootstrap_samples,
                                            confidence=confidence, seed=self.seed + epoch,
                                            allow_selection=True)
                    h_upper = audit["metrics"]["regression"]["upper"]
                    c_upper = audit["metrics"]["extra_compute"]["upper"]
                    alpha = max(0., alpha + dual_rate * (h_upper - risk_budget))
                    beta = max(0., beta + dual_rate * (c_upper - compute_budget))
                    feasible = h_upper <= risk_budget and c_upper <= compute_budget
                    value = audit["metrics"]["improvement"]["estimate"]
                    info.update(risk_upper=h_upper, compute_upper=c_upper, feasible=feasible,
                                selection_improvement=value)
                    if feasible and value > best_value:
                        best_value, best = value, self.network.state()
                history.append(info)
        if constrained:
            if best is not None:
                self.network = _MLP.restore(best)
            else:
                # Explicit all-silent fallback; feasibility is still reported,
                # since unavoidable c0 can itself exceed the declared budget.
                self.force_silent = True
        self.report = {"method": "constrained_direct_policy" if constrained else "direct_policy",
                       "training_hash": self.training_hash, "history": history,
                       "constraint_data_hash": digest(constraint_rows) if constraint_rows else None,
                       "feasible_checkpoint_found": best is not None if constrained else None,
                       "fallback": "silent" if getattr(self, "force_silent", False) else None,
                       "optimization_config": {"epochs": epochs, "learning_rate": learning_rate,
                           "kl_weight": kl_weight, "risk_budget": risk_budget, "compute_budget": compute_budget,
                           "dual_rate": dual_rate, "audit_every": audit_every,
                           "bootstrap_samples": bootstrap_samples, "confidence": confidence},
                       "final_dual_risk": alpha, "final_dual_cost": beta,
                       "uncertainty": "held-out cluster bootstrap audit; no trajectory guarantee"}
        self.selection_rows = _lineage(constraint_rows) if constraint_rows else []
        return self.report

    def fit_paired(self, rows, selection_rows=None, **kwargs):
        _validate_rows(rows, "paired")
        return self._optimize(rows, [r["D"] for r in rows], constraint_rows=selection_rows, **kwargs)

    def fit_dr(self, rows, crossfit, selection_rows=None, *, clipping=None, constrained=False, **kwargs):
        _validate_rows(rows, "single")
        _validate_crossfit(rows, crossfit)
        pseudo = dr_pseudo_outcomes(rows, crossfit["mu"], clipping=clipping)
        report = self._optimize(rows, pseudo[:, 1] - pseudo[:, 0], constraint_rows=selection_rows,
                                constrained=constrained, **kwargs)
        report.update(estimator="logged_single_arm_DR", crossfit_hash=digest(crossfit), clipping=clipping,
                      paired_harm_identified=False,
                      risk_constraint_source="separate_paired_selection" if constrained else "not_identified")
        return report

    def save(self, path):
        if self.network is None:
            raise ValueError("cannot save uninitialized policy")
        payload = {"hidden": self.hidden, "seed": self.seed, "schema": self.schema,
                   "mean": self.mean.tolist(), "scale": self.scale.tolist(), "network": self.network.state(),
                   "reference_state": self.reference_state, "training_hash": self.training_hash,
                   "training_rows": self.training_rows, "report": self.report, "safety": self.safety,
                   "selection_rows": getattr(self, "selection_rows", []),
                   "force_silent": getattr(self, "force_silent", False)}
        if self.outcome_model is not None:
            # Embed the outcome artifact so a policy is independently loadable.
            model = self.outcome_model
            payload["outcome"] = {"hidden": model.hidden, "ensemble_size": model.ensemble_size, "seed": model.seed,
                "schema": model.schema, "mean": model.mean.tolist(), "scale": model.scale.tolist(),
                "target_offset": model.target_offset.tolist(), "target_scale": model.target_scale.tolist(),
                "networks": [n.state() for n in model.networks], "training_hash": model.training_hash,
                "training_rows": model.training_rows, "training_groups": model.training_groups,
                "fit_config": model.fit_config,
                "calibration": model.calibration, "calibration_rows": getattr(model, "calibration_rows", [])}
        _checkpoint("binary_policy", payload, path)

    @classmethod
    def load(cls, path):
        value = _load(path, "binary_policy")
        obj = cls(value["hidden"], value["seed"])
        for key in ("schema", "reference_state", "training_hash", "training_rows", "report", "safety", "force_silent", "selection_rows"):
            setattr(obj, key, value[key])
        obj.mean, obj.scale = np.asarray(value["mean"]), np.asarray(value["scale"])
        obj.network = _MLP.restore(value["network"])
        if "outcome" in value:
            m = value["outcome"]
            model = OutcomeModel(m["hidden"], m["ensemble_size"], m["seed"])
            for key in ("schema", "training_hash", "training_rows", "training_groups", "calibration", "calibration_rows"):
                setattr(model, key, m[key])
            model.mean, model.scale = np.asarray(m["mean"]), np.asarray(m["scale"])
            model.fit_config = m.get("fit_config", {})
            model.target_offset, model.target_scale = np.asarray(m.get("target_offset", [0.] * 4)), np.asarray(m.get("target_scale", [1.] * 4))
            model.networks = [_MLP.restore(n) for n in m["networks"]]
            obj.outcome_model = model
        return obj


def cross_fit_rewards(rows, *, folds=3, hidden=16, seed=42, epochs=250, learning_rate=.01):
    """Group cross-fitting of separate arm reward models, no risk labels invented.

    A prediction for a row is produced without its group/task/checkpoint in the
    nuisance training set. This does not make subsequent policy training and
    evaluation on the same rows an honest held-out policy evaluation.
    """
    _validate_rows(rows, "single")
    roles = {r.get("split", "train") for r in rows}
    if len(roles) != 1 or not roles <= {"train", "audit", "test"}:
        raise ValueError("cross-fitting accepts one explicit train, audit, or test cohort without mixing")
    groups = sorted({r["group_id"] for r in rows})
    if type(folds) is not int or not 2 <= folds <= len(groups):
        raise ValueError("cross-fitting needs 2..number-of-independent-groups folds")
    shuffled = np.asarray(groups)[np.random.default_rng(seed).permutation(len(groups))]
    assignment = {str(g): i % folds for i, g in enumerate(shuffled)}
    mu = np.empty((len(rows), 2))
    fold_reports = []
    schema = _schema(rows)
    for fold in range(folds):
        train_indices = [i for i, r in enumerate(rows) if assignment[r["group_id"]] != fold]
        held_indices = [i for i, r in enumerate(rows) if assignment[r["group_id"]] == fold]
        train, held = [rows[i] for i in train_indices], [rows[i] for i in held_indices]
        _disjoint(train, held, label="cross-fit folds")
        raw = _matrix(train, schema)
        mean, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
        test_x = (_matrix(held, schema) - mean) / scale
        for z in (0, 1):
            arm = [r for r in train if r["z"] == z]
            if len({r["group_id"] for r in arm}) < 2:
                raise ValueError("each nuisance arm needs at least two training groups in every fold")
            x = (_matrix(arm, schema) - mean) / scale
            target = np.asarray([[r["r"]] for r in arm])
            net = _MLP(x.shape[1], hidden, 1, seed + fold * 2 + z)
            _fit_network(net, x, target, row_weights(arm), epochs=epochs, learning_rate=learning_rate)
            mu[held_indices, z] = net.forward(test_x)[0][:, 0]
        fold_reports.append({"fold": fold, "training_groups": sorted({r["group_id"] for r in train}),
                             "held_groups": sorted({r["group_id"] for r in held})})
    return {"schema_version": "group-crossfit.v1", "rows_hash": digest(rows), "mu": mu.tolist(),
            "folds": fold_reports, "assignment": assignment, "feature_schema": schema,
            "nuisance_architecture": {"hidden": hidden, "outputs": 1, "arm_models": 2},
            "fit_config": {"epochs": epochs, "learning_rate": learning_rate, "seed": seed},
            "warning": "OOF nuisance predictions do not replace held-out frozen-policy evaluation"}


def _validate_crossfit(rows, crossfit):
    if crossfit.get("schema_version") != "group-crossfit.v1" or crossfit.get("rows_hash") != digest(rows):
        raise ValueError("cross-fit predictions do not match this cohort")
    if np.asarray(crossfit.get("mu")).shape != (len(rows), 2):
        raise ValueError("cross-fit nuisance predictions must be n by 2")
    for fold in crossfit["folds"]:
        if set(fold["training_groups"]) & set(fold["held_groups"]):
            raise ValueError("cross-fit group leakage")


def dr_pseudo_outcomes(rows, mu, *, clipping=None):
    _validate_rows(rows, "single")
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (len(rows), 2) or not np.isfinite(mu).all():
        raise ValueError("nuisance predictions must be finite n by 2")
    if clipping is not None and (not np.isfinite(clipping) or clipping <= 0):
        raise ValueError("clipping must be positive; None is the unclipped estimator")
    output = mu.copy()
    for i, row in enumerate(rows):
        weight = 1 / row["b"]
        if clipping is not None:
            weight = min(weight, clipping)
        output[i, row["z"]] += weight * (row["r"] - mu[i, row["z"]])
    return output
