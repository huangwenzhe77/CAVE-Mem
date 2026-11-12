"""End-to-end scope checks for a frozen source bank shared across target splits."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cave_mem_v2.contracts import StudySpec, digest
from cave_mem_v2.data import build_dataset, build_split_manifest
from cave_mem_v2.environments import FactorialEnvironment
from cave_mem_v2.evaluation import evaluate_deployment
from cave_mem_v2.learning import BinaryPolicy, OutcomeModel
from cave_mem_v2.memory import FixedRetriever
from cave_mem_v2.providers import ScriptedActor
from cave_mem_v2.runtime import InterventionCollector
from cave_mem_v2.workflow import VersionedSelector
from test_source_data import verified_memory


class SharedBankAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = verified_memory("shared-bank-audit-source")
        cls.spec = StudySpec(replicates=1)
        cls.collector = InterventionCollector(cls.spec)
        cls.selector = VersionedSelector(FixedRetriever([cls.memory]))
        cls.factories, prepared, assignments = {}, [], {}
        splits = ["train"] * 4 + ["calibration"] * 2 + ["selection"] * 2 + ["audit"] * 2 + ["test"] * 2
        for index, split in enumerate(splits):
            trial_id, task_id = f"shared-trial-{index}", f"shared-target-{index}"
            factory = lambda task_id=task_id, index=index: FactorialEnvironment(
                task_id=task_id, applicable=index % 2 == 0, safe_intent=False)
            cls.factories[trial_id] = factory
            env = factory()
            env.reset()
            prepared.append(cls.collector.prepare(env, ScriptedActor(), cls.selector, trial_id=trial_id))
            assignments[trial_id] = split
        cls.trials = [item.trial for item in prepared]
        cls.manifest = build_split_manifest(cls.trials, source_disjoint=False, assignments=assignments)
        records = [cls.collector.collect_prepared_pair(item, cls.factories[item.trial["trial_id"]],
                   ScriptedActor, replicate=0) for item in prepared]
        if not all(record["valid"] for record in records):
            raise AssertionError([record.get("error") for record in records if not record["valid"]])
        cls.dataset = build_dataset(records, manifest=cls.manifest)
        cls.rows = cls.dataset["partitions"]
        cls.model = OutcomeModel(hidden=4, ensemble_size=2).fit(cls.rows["train"], epochs=3)
        cls.model.calibrate(cls.rows["calibration"], epochs=3)
        cls.policy = BinaryPolicy(hidden=4).warm_start(cls.rows["train"], cls.model, epochs=3)

    def deployment_records(self, *, trial_id_override=None):
        records = []
        for row in self.rows["test"]:
            env = self.factories[row["trial_id"]]()
            env.reset()
            record = self.collector.collect_deployment(env, ScriptedActor(), self.selector, self.policy,
                trial_id=trial_id_override or row["trial_id"])
            self.assertTrue(record["valid"], record.get("error"))
            record["evaluation_split"] = "test"
            record["evaluation_group_id"] = row["group_id"]
            record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
            records.append(record)
        return records

    def test_scope_survives_dataset_training_and_checkpoint_roundtrip(self):
        self.assertFalse(self.manifest["source_disjoint"])
        self.assertFalse(self.dataset["summary"]["source_disjoint"])
        for rows in self.rows.values():
            for row in rows:
                self.assertIs(row["source_disjoint"], False)
                self.assertFalse(any(k.startswith(("source_task:", "source_trace:", "memory:"))
                                     for k in row["group_keys"]))
        with tempfile.TemporaryDirectory() as tmp:
            model_path, policy_path = Path(tmp) / "model.json", Path(tmp) / "policy.json"
            self.model.save(model_path)
            self.policy.save(policy_path)
            loaded_model, loaded_policy = OutcomeModel.load(model_path), BinaryPolicy.load(policy_path)
            np.testing.assert_allclose(loaded_policy.predict_probability(self.rows["test"]),
                                       self.policy.predict_probability(self.rows["test"]))
            self.assertTrue(all(r["source_disjoint"] is False for r in loaded_model.training_rows))
            self.assertTrue(all(r["source_disjoint"] is False for r in loaded_policy.training_rows))
            report = evaluate_deployment(self.deployment_records(), loaded_policy,
                                          source_disjoint=False, bootstrap_samples=20)
            self.assertEqual(report["completed"], 2)
            self.assertIs(report["source_disjoint"], False)
            self.assertIsNone(report["regression"])
            self.assertTrue(report["lineage_checked"])

    def test_evaluation_cannot_silently_switch_independence_scope(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            evaluate_deployment(self.deployment_records()[:1], self.policy,
                                source_disjoint=True, bootstrap_samples=20)

    def test_shared_bank_does_not_allow_reused_target_with_a_new_trial_id(self):
        training = self.rows["train"][0]
        env = self.factories[training["trial_id"]]()
        env.reset()
        record = self.collector.collect_deployment(env, ScriptedActor(), self.selector, self.policy,
                                                    trial_id="renamed-training-target")
        self.assertTrue(record["valid"], record.get("error"))
        record["evaluation_split"] = "test"
        record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_deployment([record], self.policy, source_disjoint=False, bootstrap_samples=20)

    def test_mixed_scope_training_is_rejected(self):
        rows = copy.deepcopy(self.rows["train"])
        rows[0]["source_disjoint"] = True
        with self.assertRaisesRegex(ValueError, "mixed.*scope"):
            OutcomeModel(hidden=4, ensemble_size=2).fit(rows, epochs=1)

    def test_default_source_disjoint_design_does_not_count_shared_source_as_independent(self):
        with self.assertRaisesRegex(ValueError, "independent groups"):
            build_split_manifest(self.trials)


if __name__ == "__main__":
    unittest.main()
