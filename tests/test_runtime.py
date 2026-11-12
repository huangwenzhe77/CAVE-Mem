import copy
import tempfile
import unittest
from pathlib import Path

from cave_mem_v2.contracts import FeatureRecord, SourceMemory, StudySpec, digest
from cave_mem_v2.environments import FactorialEnvironment
from cave_mem_v2.providers import ProviderError, ScriptedActor
from cave_mem_v2.runtime import InterventionCollector


def checks(context, action, memory):
    return {"v": 1, "rho": int(context.facts.get("requires_preparation", False)),
            "e": int(action.get("type") == "operate")}


def features(context, action, memory, assessment, spec):
    return FeatureRecord(tuple(float(assessment[k]) for k in ("v", "rho", "e")), ("v", "rho", "e"))


def memory():
    return SourceMemory("test-memory", "source", "trace", "source-cp", "Operating requires preparation",
                        {"type": "operate"}, {"type": "prepare"})


class RuntimeTests(unittest.TestCase):
    def collector(self, **kwargs):
        return InterventionCollector(StudySpec(**kwargs), check_builder=checks, feature_builder=features)

    def test_draft_once_cached_action_and_actual_transient_prompts(self):
        calls = []
        class RecordingActor(ScriptedActor):
            def propose(self, context, **kwargs):
                response = super().propose(context, **kwargs)
                calls.append((kwargs["stage"], kwargs.get("reminder"), copy.deepcopy(response)))
                return response
        result = self.collector().collect_pair(lambda: FactorialEnvironment(), RecordingActor,
                                              memory(), trial_id="once")
        self.assertTrue(result["valid"], result)
        self.assertEqual(sum(stage == "draft" for stage, _, _ in calls), 1)
        self.assertEqual(sum(stage == "revision" for stage, _, _ in calls), 1)
        self.assertEqual(result["arms"]["silent"]["actions"][0], result["trial"]["draft"]["action"])
        self.assertEqual(result["labels"]["rescue"], 1)
        for stage, reminder, response in calls:
            if stage == "continuation":
                self.assertIsNone(reminder)
                self.assertNotIn("Temporary source-grounded reminder", str(response.messages))

    def test_budget_competition_has_an_explicit_separate_protocol(self):
        factory = lambda: FactorialEnvironment(safe_intent=True, max_steps=4)
        total = self.collector(max_steps=4, max_model_calls=3, budget_mode="total").collect_pair(
            factory, ScriptedActor, memory(), trial_id="total")
        continuation = self.collector(max_steps=4, max_model_calls=3, budget_mode="continuation").collect_pair(
            factory, ScriptedActor, memory(), trial_id="continuation")
        self.assertTrue(total["valid"], total)
        self.assertTrue(total["arms"]["silent"]["success"])
        self.assertFalse(total["arms"]["remind"]["success"])
        self.assertTrue(continuation["arms"]["silent"]["success"])
        self.assertTrue(continuation["arms"]["remind"]["success"])

    def test_tool_exception_does_not_become_failure_label(self):
        class BrokenEnv(FactorialEnvironment):
            def step(self, action, seed=0):
                raise RuntimeError("tool infrastructure error")
        result = self.collector().collect_pair(BrokenEnv, ScriptedActor, memory(), trial_id="broken")
        self.assertFalse(result["valid"])
        self.assertNotIn("labels", result)
        self.assertTrue(all(a["success"] is None for a in result["arms"].values()))

    def test_restoration_mismatch_is_rejected(self):
        class MismatchedEnv(FactorialEnvironment):
            def restore(self, checkpoint):
                super().restore(checkpoint)
                self._state["prepared"] = not self._state["prepared"]
                return self.observe()
        result = self.collector().collect_pair(MismatchedEnv, ScriptedActor, memory(), trial_id="mismatch")
        self.assertFalse(result["valid"])
        self.assertNotIn("labels", result)

    def test_single_logs_known_probability_without_counterfactual_labels(self):
        env = FactorialEnvironment()
        record = self.collector(reminder_probability=.3).collect_single(
            env, ScriptedActor(), memory(), trial_id="single", reset=True)
        self.assertTrue(record["valid"], record)
        self.assertIn(record["observed"]["z"], (0, 1))
        expected = .3 if record["observed"]["z"] else .7
        self.assertAlmostEqual(record["observed"]["propensity"], expected)
        self.assertEqual(len(record["arms"]), 1)
        self.assertNotIn("labels", record)
        self.assertNotIn("H", record.get("observed", {}))

    def test_private_snapshot_controls_do_not_enter_trial(self):
        result = self.collector().collect_pair(FactorialEnvironment, ScriptedActor, memory(), trial_id="private")
        self.assertTrue(result["valid"], result)
        self.assertNotIn("controls", result["trial"])
        self.assertNotIn("checkpoint", result["trial"])
        self.assertNotIn("actor_profile", str(result["trial"]["features"]))
        self.assertIn("feature_hash", result)
        self.assertEqual(result["feature_hash"], digest(result["trial"]["features"]))


if __name__ == "__main__":
    unittest.main()
