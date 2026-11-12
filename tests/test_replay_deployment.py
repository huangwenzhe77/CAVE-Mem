import copy
import tempfile
import unittest
from pathlib import Path

from cave_mem_v2.alfworld import AlfworldEnvironment
from cave_mem_v2.baselines import RulePolicy
from cave_mem_v2.contracts import StudySpec, digest
from cave_mem_v2.evaluation import evaluate_deployment
from cave_mem_v2.features import FeatureBuilder
from cave_mem_v2.providers import ScriptedActor
from cave_mem_v2.runtime import InterventionCollector, ProtocolError
from cave_mem_v2.environments import FactorialEnvironment
from test_runtime import checks, features, memory


class ReplayDeploymentTests(unittest.TestCase):
    def test_replicates_share_one_draft_and_resume_does_no_new_work(self):
        calls, order = [], []
        class Actor(ScriptedActor):
            def propose(self, context, **kwargs):
                calls.append(kwargs["stage"])
                order.append(kwargs["stage"])
                return super().propose(context, **kwargs)
        class Selector:
            version = "test-frozen-bank-1"
            def __call__(self, context):
                order.append("retrieve")
                return memory(), .7
        with tempfile.TemporaryDirectory() as path:
            kwargs = dict(check_builder=checks, feature_builder=features, artifact_dir=path)
            first = InterventionCollector(StudySpec(replicates=3), **kwargs)
            rows = first.collect_case(FactorialEnvironment, Actor, Selector(), trial_id="reps")
            self.assertTrue(all(r["valid"] for r in rows))
            self.assertEqual(calls.count("draft"), 1)
            self.assertEqual(order[:2], ["draft", "retrieve"])
            self.assertEqual(len({r["trial"]["draft"]["action_hash"] for r in rows}), 1)
            before = list(calls)
            second = InterventionCollector(StudySpec(replicates=3), **kwargs)
            resumed = second.collect_case(FactorialEnvironment, Actor, Selector(), trial_id="reps")
            self.assertEqual(rows, resumed)
            self.assertEqual(calls, before)

    def test_prepared_nested_mutation_is_detected(self):
        collector = InterventionCollector(check_builder=checks, feature_builder=features)
        env = FactorialEnvironment()
        env.reset()
        prepared = collector.prepare(env, ScriptedActor(), memory(), trial_id="mutate")
        prepared.trial["features"]["values"][0] = -100
        with self.assertRaisesRegex(ProtocolError, "mutat"):
            collector.collect_prepared_pair(prepared, FactorialEnvironment, ScriptedActor)

    def test_actual_policy_rollouts_are_not_randomized_training_logs(self):
        collector = InterventionCollector(StudySpec(replicates=1), check_builder=checks, feature_builder=features)
        policy = RulePolicy("silent")
        records = []
        for i in range(3):
            env = FactorialEnvironment(task_id=f"rollout-{i}", applicable=False)
            env.reset()
            m = memory()
            m = type(m)(**{**m.__dict__, "source_task_id": f"source-{i}", "source_trace_id": f"trace-{i}",
                           "memory_id": f"memory-{i}"})
            record = collector.collect_deployment(env, ScriptedActor(), m, policy, trial_id=f"live-{i}")
            self.assertTrue(record["valid"], record.get("error"))
            self.assertEqual(record["schema_version"], "deployment-intervention.v2")
            self.assertEqual(set(record["arms"]), {"silent"})
            self.assertNotIn("assignment", record)
            self.assertEqual(record["decision"]["probability"], 0.)
            records.append(record)
        report = evaluate_deployment(records, policy, bootstrap_samples=30)
        self.assertEqual(report["metrics"]["success"]["estimate"], 1.)
        self.assertFalse(report["paired_harm_identified"])
        self.assertIsNone(report["regression"])

        # A changed target ID cannot hide a shared training source lineage.
        class FittedSilent:
            training_rows = [{"task_id": "other-task", "checkpoint_id": "other-cp",
                              "trial_id": "other-trial", "group_id": "other-group",
                              "group_keys": ["source_task:source-0"]}]
            def predict_probability(self, rows):
                return [0.] * len(rows)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_deployment(records, FittedSilent(), bootstrap_samples=30)

    def test_annotations_cannot_split_one_provenance_component(self):
        collector = InterventionCollector(StudySpec(replicates=1), check_builder=checks, feature_builder=features)
        policy = RulePolicy("silent")
        records = []
        for i in range(2):
            env = FactorialEnvironment(task_id=f"shared-{i}")
            env.reset()
            record = collector.collect_deployment(env, ScriptedActor(), memory(), policy, trial_id=f"shared-{i}")
            record["evaluation_group_id"] = f"falsely-independent-{i}"
            record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
            records.append(record)
        with self.assertRaisesRegex(ValueError, "split one connected"):
            evaluate_deployment(records, policy, bootstrap_samples=30)

    def test_deployment_evaluation_preserves_replicate_identity(self):
        collector = InterventionCollector(StudySpec(replicates=2), check_builder=checks, feature_builder=features)
        class SchemaCheckingSilent(RulePolicy):
            def predict_probability(self, rows):
                from cave_mem_v2.learning import _validate_rows
                _validate_rows(rows)
                return super().predict_probability(rows)
        policy, records = SchemaCheckingSilent("silent"), []
        for i in range(3):
            for replicate in range(2):
                env = FactorialEnvironment(task_id=f"rep-target-{i}")
                env.reset()
                m = memory()
                m = type(m)(**{**m.__dict__, "source_task_id": f"ind-source-{i}",
                               "source_trace_id": f"ind-trace-{i}", "memory_id": f"ind-memory-{i}"})
                records.append(collector.collect_deployment(env, ScriptedActor(), m, policy,
                    trial_id=f"rep-trial-{i}", replicate=replicate))
        report = evaluate_deployment(records, policy, bootstrap_samples=30)
        self.assertEqual(report["completed"], 6)
        self.assertEqual(report["metrics"]["success"]["independent_groups"], 3)

    def test_alfworld_offline_prefix_replay_stays_uncertified(self):
        class FakeWorker:
            def seed(self, value):
                self.seed_value = value
            def reset(self):
                self.n = 0
                return ["Your task is to: finish the task\nroom"], {
                    "admissible_commands": [["look"]], "won": [False], "expert_plan": [["hidden"]]}
            def step(self, actions):
                self.n += 1
                return [f"room{self.n}"], [0.], [False], {"admissible_commands": [["look"]], "won": [False]}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as path:
            asset = Path(path) / "game.tw-pddl"
            asset.write_text("fake fixture asset", encoding="utf-8")
            env = AlfworldEnvironment(gamefile=asset, env_factory=FakeWorker)
            context = env.reset(17)
            self.assertNotIn("hidden", str(context))
            env.step({"type": "alfworld_text", "text": "look"})
            checkpoint = env.snapshot()
            env.restore(checkpoint)
            self.assertEqual(digest(checkpoint), digest(env.snapshot()))
            report = env.restore_report(checkpoint)
            self.assertTrue(report["observable_replays_matched"])
            self.assertFalse(report["latent_restore_certified"])
            FeatureBuilder().build(env.observe(), {"type": "alfworld_text", "text": "look"}, None)
            self.assertFalse(env.capabilities()["restore_certified"])
            env.close()


if __name__ == "__main__":
    unittest.main()
