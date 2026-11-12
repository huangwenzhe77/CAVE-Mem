"""Complete workflow over tiny injected task assets, never real benchmark claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from cave_mem_v2.contracts import canonical, digest, read_json, read_jsonl, write_json
from cave_mem_v2.providers import ActorResponse
from cave_mem_v2.terminal import PROTOCOL, TerminalEnvironment, task_from_directory, protocol_hash
from cave_mem_v2.terminal_workflow import (
    _planned_cohort, pilot_terminal, build_terminal_sources, run_terminal_pipeline,
    rollout_terminal,
)


class ArtifactBackend:
    identity = "workflow-injected-artifact-backend.v1"
    filesystem_boundaries = True
    base = "sha256:" + "a" * 64

    def __init__(self):
        self.images = {self.base: set()}

    def ensure_image(self, task, protocol):
        task.verify_assets()
        return self.base

    def image_available(self, image):
        return image in self.images

    def execute(self, image, command, protocol, *, seed=0):
        files = set(self.images[image])
        if command == "touch bad":
            files.add("bad")
        elif command == "touch fixed":
            files.add("fixed")
        else:
            raise ValueError("unexpected injected command: " + command)
        image = "sha256:" + digest(sorted(files))
        self.images[image] = files
        return image, {"exit_code": 0, "stdout": "public command output", "stderr": ""}

    def verify(self, image, task, protocol):
        success = "fixed" in self.images[image]
        return {"success": success, "rewards": {"reward": int(success)},
                "private_test_evidence": "PRIVATE-VERIFIER-SENTINEL"}


class ArtifactActor:
    recorded_messages = []
    def configuration(self):
        return {"actor": "workflow-injected-artifact-actor.v1"}

    def capabilities(self):
        return {"stateless": True, "sampling_coupling": "deterministic"}

    def propose(self, context, *, stage, seed, reminder=None, call_id=""):
        if reminder:
            action = {"type": "shell", "command": "touch fixed"}
        elif not context.history:
            action = {"type": "shell", "command": "touch bad"}
        else:
            action = {"type": "finish"}
        messages = ({"role": "user", "content": context.instruction + (reminder or "")},)
        self.recorded_messages.append(messages)
        return ActorResponse(action, canonical(action),
            {"model_calls": 1., "physical_attempts": 1., "input_tokens": 1., "output_tokens": 1., "api_cost": 0.},
            digest(messages), messages, (), "injected")


def workflow_fixture(root, *, checkpoint_steps=0):
    root = Path(root)
    task_ids = ["source"] + [f"target-{i}" for i in range(12)]
    manifest_tasks = []
    for task_id in task_ids:
        folder = root / "tasks" / task_id
        (folder / "environment").mkdir(parents=True)
        (folder / "tests").mkdir()
        (folder / "instruction.md").write_text("Create public fixed artifact for " + task_id, encoding="utf-8")
        (folder / "task.toml").write_text('version="1.0"\n', encoding="utf-8")
        (folder / "environment/Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        (folder / "tests/test.sh").write_text("# private injected verifier\n", encoding="utf-8")
        protocol = {"max_steps": 3 + checkpoint_steps, "paired_scope": PROTOCOL,
            "exposure_rules": {"artifact_repair": {
                "trigger_actions": [{"type": "shell", "command": "touch bad"}],
                "safe_actions": [{"type": "shell", "command": "touch fixed"}]}}}
        task = task_from_directory(folder, repository_commit="b" * 40, protocol=protocol)
        # This is an explicitly injected fixture certificate, not an audit of real tasks.
        protocol["audit"] = {"asset_hash": task.asset_hash, "scope": PROTOCOL,
            "protocol_hash": protocol_hash(protocol), "repository_commit": "b" * 40,
            "backend": ArtifactBackend.identity, "base_image": ArtifactBackend.base,
            "reviewer": "unit-test-fixture", "review_notes": "injected immutable sets only",
            "reviewed_constraints": {"no_external_state": True, "no_persistent_services": True,
                "filesystem_boundary_semantics": True, "clock_and_rng_are_exogenous": True},
            "restoration_checks": [{"check_id": str(i), "snapshot_hash": "fixture", "restored_hash": "fixture"} for i in range(2)]}
        manifest_tasks.append({"task_id": task_id, "path": task_id, "asset_hash": task.asset_hash, "protocol": protocol})
    manifest = root / "manifest.json"
    write_json(manifest, {"schema_version": "terminal-task-manifest.v1", "benchmark": "terminal-bench@2.0",
        "repository_commit": "b" * 40, "root": str(root / "tasks"), "tasks": manifest_tasks})
    config = root / "study.toml"
    splits = ["train"] * 4 + ["calibration"] * 2 + ["selection"] * 2 + ["audit"] * 2 + ["test"] * 2
    config.write_text('\n'.join([
        '[study]', 'study_id="injected-terminal-workflow"', 'max_steps=3', 'max_model_calls=4',
        'replicates=2', 'lambda_inference=0.01', 'risk_budget=1.0', 'compute_budget=10.0',
        '[training]', 'hidden_size=4', 'epochs=8', '[terminal]', 'source_ids=["source"]',
        'target_ids=' + json.dumps(task_ids[1:]), 'source_disjoint=false', f'checkpoint_steps={checkpoint_steps}',
        '[terminal.assignments]', *[f'"target-{i}"="{split}"' for i, split in enumerate(splits)],
        '[source]', 'screening_pairs=2', 'confirmation_pairs=16', 'min_confirmation_pairs=16',
    ]) + '\n', encoding="utf-8")
    backend = ArtifactBackend()
    return config, manifest, backend


class TerminalWorkflowTests(unittest.TestCase):
    def test_full_source_confirmation_collection_training_actual_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ArtifactActor.recorded_messages.clear()
            config, manifest, backend = workflow_fixture(root)
            kwargs = {"make_actor": ArtifactActor,
                      "make_env": lambda task: TerminalEnvironment(task, backend=backend)}
            pilot_terminal(config, manifest, root / "pilot", **kwargs)
            pilot = read_json(root / "pilot/pilot.json")
            trace_id = pilot["traces"][0]["trace_id"]
            reviewed = read_json(root / "pilot/candidates.json")
            reviewed["candidates"] = [{"trace_id": trace_id, "step_index": 0, "approved": True,
                "problem_action": {"type": "shell", "command": "touch bad"},
                "correction": {"type": "shell", "command": "touch fixed"},
                "mechanism": "artifact_repair", "positive_conditions": [], "exclusion_conditions": []}]
            write_json(root / "reviewed.json", reviewed)
            source = build_terminal_sources(config, manifest, root / "pilot", root / "reviewed.json", root / "sources", **kwargs)
            self.assertEqual(source["qualified_memories"], 1)
            source_replays = next((root / "sources/source-replays").glob("*.confirmation.jsonl"))
            source_arm = read_jsonl(source_replays)[0]["arms"]["corrected"]
            self.assertEqual(source_arm["private_verifier"]["report"]["private_test_evidence"], "PRIVATE-VERIFIER-SENTINEL")
            self.assertEqual(source_arm["private_verifier"]["sha256"], digest(source_arm["private_verifier"]["report"]))
            trace = read_json(root / "pilot/traces" / (trace_id + ".json"))
            self.assertTrue((root / "pilot" / trace["private_verifier_reference"]["path"]).is_file())
            result = run_terminal_pipeline(config, manifest, root / "sources", root / "run", **kwargs)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["execution_backend"], "injected_test_backend")
            self.assertFalse(result["official_harbor_session_equivalence"])
            no_memory = read_json(root / "run/no-memory-rollout/report.json")
            always = read_json(root / "run/baseline-rollouts/always_remind/report.json")
            silent = read_json(root / "run/baseline-rollouts/silent/report.json")
            self.assertEqual(no_memory["completed"], 4)
            self.assertEqual(no_memory["metrics"]["success"]["estimate"], 0.)
            self.assertEqual(no_memory["metrics"]["cost_gate_calls"]["estimate"], 0.)
            self.assertEqual(no_memory["metrics"]["cost_retrieval_calls"]["estimate"], 0.)
            self.assertEqual(always["metrics"]["success"]["estimate"], 1.)
            self.assertEqual(always["metrics"]["intervention_rate"]["estimate"], 1.)
            self.assertEqual(silent["metrics"]["cost_gate_calls"]["estimate"], 1.)
            self.assertEqual(silent["metrics"]["cost_retrieval_calls"]["estimate"], 1.)
            rows = read_jsonl(root / "run/dataset/test.jsonl")
            self.assertTrue(all(row["source_disjoint"] is False for row in rows))
            self.assertNotIn("PRIVATE-VERIFIER-SENTINEL", canonical(rows))
            self.assertNotIn("PRIVATE-VERIFIER-SENTINEL", canonical(ArtifactActor.recorded_messages))
            deployed = read_json(next((root / "run/baseline-rollouts/always_remind/annotated").glob("*.json")))
            self.assertEqual(deployed["arms"]["remind"]["private_verifier"]["report"]["private_test_evidence"], "PRIVATE-VERIFIER-SENTINEL")
            self.assertIn("injected", read_json(root / "run/test-report.json")["role"])

            class BrokenPrefix(TerminalEnvironment):
                def reset(self, seed=0):
                    raise RuntimeError("injected prefix infrastructure failure")

            failed = rollout_terminal(config, manifest, root / "sources", root / "run/collection",
                root / "run/dataset", root / "run/models", root / "failed", policy_file=None,
                make_actor=ArtifactActor, make_env=lambda task: BrokenPrefix(task, backend=backend))
            self.assertEqual((failed["planned"], failed["completed"]), (4, 0))
            self.assertEqual(len(failed["excluded"]), 4)
            self.assertEqual(failed["metrics"], {})

    def test_small_predeclared_cohort_is_rejected_without_actor(self):
        raw = {"terminal": {"target_ids": [f"task-{i}" for i in range(5)], "source_disjoint": False}}
        with self.assertRaisesRegex(ValueError, "insufficient independent groups"):
            _planned_cohort(raw, {"config_hash": "fixture", "manifest_hash": "fixture"})


if __name__ == "__main__":
    unittest.main()
