import copy
import tempfile
import unittest
from dataclasses import replace

from cave_mem_v2.contracts import PublicContext, SourceMemory, StudySpec, digest, plain
from cave_mem_v2.data import build_dataset, build_split_manifest
from cave_mem_v2.environments import FactorialEnvironment
from cave_mem_v2.features import FeatureBuilder, FrozenEncoder, assess
from cave_mem_v2.memory import FixedRetriever, MemoryBank
from cave_mem_v2.providers import ScriptedActor
from cave_mem_v2.runtime import InterventionCollector
from cave_mem_v2.source import EvidenceSpec, build_memory, evaluate_evidence, replay_source, validate_evidence


def verified_memory(source_id="source-test"):
    factory = lambda: FactorialEnvironment(task_id=source_id)
    env = factory()
    env.reset()
    checkpoint = env.snapshot()
    spec = EvidenceSpec(min_confirmation_pairs=16)
    rows = replay_source(factory, ScriptedActor, checkpoint, {"type": "operate"},
                         {"type": "prepare"}, replicates=16, spec=spec)
    evidence = evaluate_evidence(rows, spec=spec)
    return build_memory(source_task_id=source_id, source_trace_id=source_id + ":trace",
                        source_checkpoint_hash=digest(checkpoint), mechanism="required preparation",
                        problem_action={"type": "operate"}, correction={"type": "prepare"},
                        positive_conditions=({"field": "requires_preparation", "equals": True},), evidence=evidence)


class SourceFeatureDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = verified_memory()

    def test_source_certificate_binds_actions_checkpoint_and_declared_claim(self):
        self.assertEqual(validate_evidence(self.memory)["v"], 1)
        for edited in (replace(self.memory, correction={"type": "finish"}),
                       replace(self.memory, problem_action={"type": "prepare"}),
                       replace(self.memory, source_checkpoint_hash="different"),
                       replace(self.memory, positive_conditions=()),
                       replace(self.memory, mechanism="an untested new claim")):
            self.assertEqual(validate_evidence(edited)["v"], 0)
        bad = copy.deepcopy(self.memory.evidence)
        bad["lower_bound"] = 100
        self.assertEqual(validate_evidence(replace(self.memory, evidence=bad))["v"], 0)

    def test_confirmation_cannot_reuse_screening_or_count_failed_replay_as_zero(self):
        evidence = self.memory.evidence
        spec = EvidenceSpec(**evidence["spec"])
        rows = copy.deepcopy(evidence["confirmation"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_evidence(rows, spec=spec, screening_ids=(rows[0]["sample_id"],))
        rows[0]["valid"] = False
        rows[0]["record_hash"] = digest({k: v for k, v in rows[0].items() if k != "record_hash"})
        report = evaluate_evidence(rows, spec=spec)
        self.assertEqual(report["v"], "?")
        self.assertIn("invalid", report["reason"])
        self.assertTrue(all("costs" in a for r in evidence["confirmation"] for a in r["arms"].values()))

    def test_target_predicates_and_cached_action_are_independent_axes(self):
        for applicable in (False, True):
            for safe in (False, True):
                env = FactorialEnvironment(applicable=applicable, safe_intent=safe)
                context = env.reset()
                draft = {"type": "prepare" if safe else "operate"}
                checks = assess(context, draft, self.memory)
                self.assertEqual((checks["v"], checks["rho"], checks["e"]), (1, int(applicable), int(not safe)))
        env = FactorialEnvironment(hide_applicability=True)
        self.assertEqual(assess(env.reset(), {"type": "operate"}, self.memory)["rho"], "?")

    def test_private_metadata_is_rejected_and_ids_do_not_change_features(self):
        env = FactorialEnvironment()
        context = env.reset()
        f = FeatureBuilder()
        first = f.build(context, {"type": "operate"}, self.memory)
        second = f.build(replace(context, task_id="different-id"), {"type": "operate"}, self.memory)
        self.assertEqual(first, second)
        bad = replace(context, facts={**context.facts, "actor_profile": "success_label"})
        with self.assertRaisesRegex(ValueError, "unapproved"):
            f.build(bad, {"type": "operate"}, self.memory)
        with self.assertRaises(ValueError):
            FeatureBuilder(allowed_fact_names={"actor_profile"})

    def test_encoder_cache_revision_dimension_and_nonfinite(self):
        calls = []
        def encode(texts):
            calls.append(list(texts))
            return [[float(len(t)), 1.] for t in texts]
        with tempfile.TemporaryDirectory() as path:
            enc = FrozenEncoder(encode, version="pinned-weights-1", cache_dir=path)
            enc.encode(["abc", "abc"])
            self.assertEqual(calls, [["abc"]])
            FrozenEncoder(encode, version="pinned-weights-1", cache_dir=path).encode(["abc"])
            self.assertEqual(len(calls), 1)
            FrozenEncoder(encode, version="pinned-weights-2", cache_dir=path).encode(["abc"])
            self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            FrozenEncoder(lambda _: [[float("nan")]], version="bad").encode(["x"])

    def test_bank_conflicts_and_retriever_freeze_are_not_silent(self):
        bank = MemoryBank(memories=[self.memory])
        retriever = FixedRetriever(bank)
        bank.put(replace(self.memory, memory_id="new", mechanism="another"))
        env = FactorialEnvironment()
        selected, score = retriever.retrieve(env.reset())
        self.assertEqual(selected.content_hash(), self.memory.content_hash())
        with self.assertRaisesRegex(ValueError, "conflict"):
            bank.put(replace(self.memory, mechanism="changed"))

    def test_shared_source_links_trials_unless_design_explicitly_changes(self):
        trials = [{"trial_id": f"t{i}", "task_id": f"task{i}", "checkpoint_id": f"cp{i}",
                   "memory": plain(self.memory)} for i in range(10)]
        with self.assertRaisesRegex(ValueError, "independent groups"):
            build_split_manifest(trials)
        manifest = build_split_manifest(trials, source_disjoint=False)
        self.assertFalse(manifest["source_disjoint"])
        self.assertEqual(sum(manifest["group_counts"].values()), 10)
        # The chosen semantic axis also forms indivisible components.
        semantic = {t["trial_id"]: {"mechanism": "same"} for t in trials}
        with self.assertRaisesRegex(ValueError, "independent groups"):
            build_split_manifest(trials, source_disjoint=False, semantic_metadata=semantic, holdout_axis="mechanism")

    def test_export_recomputes_labels_and_keeps_privileged_artifacts_out(self):
        collector = InterventionCollector(StudySpec(replicates=1))
        record = collector.collect_pair(lambda: FactorialEnvironment(task_id="target"), ScriptedActor,
                                         self.memory, trial_id="dataset")
        self.assertTrue(record["valid"], record.get("error"))
        manifest = build_split_manifest([record["trial"]], ratios={"train": 1.})
        dataset = build_dataset([record], manifest=manifest)
        row = dataset["partitions"]["train"][0]
        self.assertAlmostEqual(row["D"], row["r1"] - row["r0"])
        self.assertNotIn("context", row)
        self.assertNotIn("memory", row)
        self.assertNotIn("checkpoint", row)
        self.assertIn("group_keys", row)
        bad = copy.deepcopy(record)
        bad["labels"]["regression"] = 1 - bad["labels"]["regression"]
        bad["record_hash"] = digest({k: v for k, v in bad.items() if k != "record_hash"})
        rejected = build_dataset([bad], manifest=manifest)
        self.assertEqual(rejected["summary"]["completed"], 0)
        self.assertIn("regression", rejected["rejected"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
