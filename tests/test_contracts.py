"""Independent numerical and information-boundary checks."""
import math
import unittest

from cave_mem_v2.contracts import (
    FeatureRecord, PublicContext, SourceMemory, StudySpec,
    canonical, digest, stage_seed, truth, utility, weighted_cost,
)


class ContractTests(unittest.TestCase):
    def test_truth_states_are_not_python_booleans(self):
        for value in (0, 1, "?"):
            self.assertEqual(truth(value), value)
        for value in (True, False, 0.0, 1.0, "true", None):
            with self.assertRaises(ValueError):
                truth(value)

    def test_unknown_cost_is_not_free(self):
        spec = StudySpec(lambda_inference=0.1)
        self.assertIsNone(utility(True, {"model_calls": None}, spec))
        self.assertAlmostEqual(utility(True, {"model_calls": 3}, spec), 0.7)
        self.assertEqual(utility(False, {}, StudySpec(lambda_inference=0)), 0.0)
        self.assertIsNone(weighted_cost({}, {"api_cost": 1}))
        with self.assertRaises(ValueError):
            weighted_cost({"model_calls": -1}, {"model_calls": 1})

    def test_seed_streams_do_not_depend_on_arm_execution_order(self):
        first = stage_seed(42, "trial", 0, "continuation", 1)
        self.assertEqual(first, stage_seed(42, "trial", 0, "continuation", 1))
        self.assertNotEqual(first, stage_seed(42, "trial", 0, "revision", 1))
        self.assertNotEqual(first, stage_seed(42, "trial", 1, "continuation", 1))

    def test_cost_protocol_and_features_reject_bad_values(self):
        for args in ({"max_steps": True}, {"budget_mode": "legacy"},
                     {"reminder_probability": 0}, {"risk_budget": 1.2},
                     {"lambda_inference": float("nan")}):
            with self.assertRaises(ValueError):
                StudySpec(**args)
        with self.assertRaises(ValueError):
            FeatureRecord((float("inf"),), ("leak",))
        with self.assertRaises(ValueError):
            FeatureRecord((1., 2.), ("x", "x"))

    def test_copy_on_contract_creation_and_content_addressing(self):
        facts = {"nested": {"value": 1}}
        context = PublicContext("t", "Do the task", "Observation", facts)
        facts["nested"]["value"] = 7
        self.assertEqual(context.facts["nested"]["value"], 1)
        memory = SourceMemory("id", "s", "tr", "cp", "mechanism", {"type": "delete"}, {"type": "backup"})
        other = SourceMemory("another-id", "s", "tr", "cp", "mechanism", {"type": "delete"}, {"type": "backup"})
        self.assertEqual(memory.content_hash(), other.content_hash())
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            canonical({"bad": math.nan})


if __name__ == "__main__":
    unittest.main()
