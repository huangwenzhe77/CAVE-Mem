"""Analytical counterexamples for policy values, leakage and off-policy estimators."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cave_mem_v2.baselines import RulePolicy
from cave_mem_v2.evaluation import cluster_bootstrap, evaluate_logged, evaluate_paired
from cave_mem_v2.learning import BinaryPolicy, OutcomeModel, cross_fit_rewards, dr_pseudo_outcomes


def paired_rows(prefix, split, groups=4):
    rows = []
    for group in range(groups):
        for sign in (-1, 1):
            y0, y1 = int(sign < 0), int(sign > 0)
            rows.append({"x": [1., float(sign)], "feature_names": ["validity", "signal"],
                         "feature_version": "analytical.v1", "eligible": True,
                         "group_id": f"{prefix}-g{group}", "task_id": f"{prefix}-t{group}",
                         "checkpoint_id": f"{prefix}-cp{group}-{sign}",
                         "trial_id": f"{prefix}-trial{group}-{sign}", "sampling_weight": 1., "split": split,
                         "D": float(y1-y0), "H": int(y0 and not y1), "c0": .1, "delta_cost": .2,
                         "y0": y0, "y1": y1, "r0": float(y0), "r1": float(y1),
                         "checks": {"v": 1, "rho": 1, "e": int(sign > 0)}})
    return rows


def logged_rows(prefix, split, groups=8):
    rows = []
    for group in range(groups):
        for x in (-1., 1.):
            for z in (0, 1):
                reward = float((x > 0) == bool(z))
                rows.append({"x": [1., x], "feature_names": ["validity", "signal"],
                             "eligible": True, "group_id": f"{prefix}-g{group}", "task_id": f"{prefix}-t{group}",
                             "checkpoint_id": f"{prefix}-cp{group}-{x}-{z}",
                             "trial_id": f"{prefix}-trial{group}-{x}-{z}", "split": split,
                             "sampling_weight": 1., "z": z, "b": .5, "r": reward})
    return rows


class LearningTests(unittest.TestCase):
    def test_policy_mixture_and_regression_have_hand_calculated_values(self):
        rows = paired_rows("test", "test")
        half = lambda row: .25
        report = evaluate_paired(half, rows, bootstrap_samples=30)
        self.assertAlmostEqual(report["metrics"]["value"]["estimate"], .5)
        self.assertAlmostEqual(report["metrics"]["regression"]["estimate"], .125)
        self.assertAlmostEqual(report["metrics"]["rescue"]["estimate"], .125)
        self.assertAlmostEqual(report["metrics"]["extra_compute"]["estimate"], .15)

    def test_dr_residual_correction_matches_equation_eight(self):
        rows = logged_rows("dr", "train", 2)
        mu = np.zeros((len(rows), 2)) + .2
        pseudo = dr_pseudo_outcomes(rows, mu)
        for i, row in enumerate(rows):
            self.assertAlmostEqual(pseudo[i, row["z"]], .2 + (row["r"] - .2)/.5)
            self.assertAlmostEqual(pseudo[i, 1-row["z"]], .2)
        bad = copy.deepcopy(rows)
        bad[0]["H"] = 0
        with self.assertRaises(ValueError):
            dr_pseudo_outcomes(bad, mu)

    def test_group_bootstrap_does_not_count_forks_as_independent_tasks(self):
        report = cluster_bootstrap([0., 0., 1., 1.], ["a", "a", "b", "b"], samples=30)
        self.assertEqual(report["independent_groups"], 2)
        self.assertAlmostEqual(report["estimate"], .5)
        with self.assertRaises(ValueError):
            cluster_bootstrap([0., 1.], ["same", "same"], samples=30)

    def test_training_calibration_frozen_audit_and_checkpoint_roundtrip(self):
        train = paired_rows("train", "train", 6)
        calibration = paired_rows("cal", "calibration", 4)
        selection = paired_rows("select", "selection", 4)
        test = paired_rows("untouched", "test", 4)
        model = OutcomeModel(hidden=8, ensemble_size=3).fit(train, epochs=180)
        with self.assertRaises(ValueError):
            model.calibrate(train)
        model.calibrate(calibration, epochs=100)
        prediction = model.predict(test)
        self.assertLess(np.mean((prediction["tau"] - np.asarray([r["D"] for r in test]))**2), .1)
        policy = BinaryPolicy(hidden=8).warm_start(train, model, epochs=120)
        policy.fit_paired(train, selection, epochs=60, audit_every=20, bootstrap_samples=30,
                          risk_budget=.1, compute_budget=.5)
        report = evaluate_paired(policy, test, bootstrap_samples=30)
        self.assertLess(report["metrics"]["regression"]["estimate"], .1)
        with self.assertRaises(ValueError):
            evaluate_paired(policy, train, bootstrap_samples=30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"policy.json"
            policy.save(path)
            loaded = BinaryPolicy.load(path)
            np.testing.assert_allclose(loaded.predict_probability(test), policy.predict_probability(test))

    def test_cross_fit_and_single_arm_evaluation_never_claim_paired_harm(self):
        rows = logged_rows("audit", "audit")
        crossfit = cross_fit_rewards(rows, folds=2, hidden=5, epochs=60)
        for fold in crossfit["folds"]:
            self.assertFalse(set(fold["training_groups"]) & set(fold["held_groups"]))
        report = evaluate_logged(RulePolicy("silent"), rows, crossfit, bootstrap_samples=30)
        self.assertFalse(report["paired_harm_identified"])
        self.assertIsNone(report["regression"])
        self.assertGreater(report["diagnostics"]["ess"], 0)


if __name__ == "__main__":
    unittest.main()
