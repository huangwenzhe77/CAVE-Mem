"""Independent terminal contract checks; injected backends are not real TB runs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cave_mem_v2.contracts import StudySpec, canonical, digest, read_json, read_jsonl, write_json
from cave_mem_v2.memory import MemoryBank
from cave_mem_v2.providers import ActorResponse
from cave_mem_v2.runtime import InterventionCollector
from cave_mem_v2.terminal import (
    PROTOCOL, TerminalEnvironment, TerminalInfrastructureError,
    load_terminal_tasks, parse_verifier_result, task_from_directory,
)


COMMIT = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"


def task_assets(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "environment").mkdir()
    (root / "tests").mkdir()
    (root / "solution").mkdir()
    (root / "instruction.md").write_text("Create the requested artifact.", encoding="utf-8")
    (root / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
    (root / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / "tests" / "test.sh").write_text("# PRIVATE-TEST-SECRET\n", encoding="utf-8")
    (root / "solution" / "solve.sh").write_text("# PRIVATE-SOLUTION-SECRET\n", encoding="utf-8")
    return task_from_directory(root, repository_commit=COMMIT, protocol={"max_steps": 3})


def report_files(root, reward, tests, **summary_overrides):
    (root / "reward.txt").write_text(str(reward), encoding="utf-8")
    summary = {"tests": len(tests), "passed": 0, "failed": 0, "skipped": 0,
               "pending": 0, "other": 0, "start": 1.0, "stop": 2.0}
    for item in tests:
        summary[item["status"]] += 1
    summary.update(summary_overrides)
    (root / "ctrf.json").write_text(json.dumps({"results": {
        "tool": {"name": "pytest", "version": "8.4.1"},
        "summary": summary, "tests": tests}}), encoding="utf-8")


class InjectedTerminalBackend:
    identity = "independent-audit-injected-runner"
    filesystem_boundaries = True

    def __init__(self, *, verify_error=False):
        self.executions = []
        self.verifications = []
        self.verify_error = verify_error

    def ensure_image(self, task, protocol):
        task.verify_assets()
        return "sha256:" + "0" * 64

    def image_available(self, image):
        return image.startswith("sha256:")

    def execute(self, image, command, protocol, *, seed=0):
        self.executions.append((image, command, copy.deepcopy(protocol)))
        return "sha256:" + "1" * 64, {"exit_code": 7, "stdout": "ordinary output",
                                       "stderr": "recoverable command failure", "truncated": False}

    def verify(self, image, task, protocol):
        self.verifications.append(image)
        if self.verify_error:
            raise TerminalInfrastructureError("dependency installation failed before tests")
        return {"success": False, "rewards": {"reward": 0},
                "private_verifier_log": "PRIVATE-TEST-SECRET"}


class ShellThenFinishActor:
    def configuration(self):
        return {"actor": "injected-shell-then-finish"}

    def capabilities(self):
        return {"stateless": True, "sampling_coupling": "deterministic-test-actor"}

    def propose(self, context, *, stage, seed, reminder=None, call_id=""):
        action = {"type": "shell", "command": "exit 7"} if stage == "draft" else {"type": "finish"}
        return ActorResponse(action, canonical(action), {"model_calls": 1, "physical_attempts": 0,
            "input_tokens": 0, "output_tokens": 0, "api_cost": 0}, digest(context), (), (), "audit-actor")


class NeverCalledPolicy:
    name = "no-memory-audit"

    def predict_probability(self, rows):
        raise AssertionError("no-candidate path must not call the gate")


class TerminalAuditTests(unittest.TestCase):
    def test_reward_zero_without_executed_tests_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reward.txt").write_text("0", encoding="utf-8")
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)
            report_files(root, 0, [])
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)

    def test_ctrf_setup_failure_without_raw_status_is_not_task_failure(self):
        # pytest-json-ctrf 0.3.5 can omit raw_status on an initial setup failure.
        for phase in ("setup", "teardown"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report_files(root, 0, [{"name": "test_artifact", "status": "failed",
                    "message": f"The test failed in the {phase} phase due to an exception"}])
                with self.assertRaises(TerminalInfrastructureError):
                    parse_verifier_result(root)

    def test_reward_and_ctrf_must_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_files(root, 0, [{"name": "test_artifact", "status": "passed"}])
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)
            report_files(root, 1, [{"name": "test_artifact", "status": "failed",
                                   "raw_status": "call_failed"}])
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)

    def test_ctrf_counts_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_files(root, 1, [{"name": "test_artifact", "status": "passed"}], passed=17)
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)

    def test_actual_assertion_failure_remains_binary_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_files(root, 0, [{"name": "test_artifact", "status": "failed",
                "raw_status": "call_failed", "message": "The test failed in the call phase due to an assertion error"}])
            report = parse_verifier_result(root)
            self.assertIs(report["success"], False)
            self.assertEqual(report["rewards"]["reward"], 0)

    def test_manifest_content_mutation_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = task_assets(root / "task")
            manifest = {"schema_version": "terminal-task-manifest.v1", "repository_commit": COMMIT,
                "benchmark": "terminal-bench@2.0", "root": str(root), "tasks": [{
                    "task_id": "task", "path": "task", "asset_hash": task.asset_hash}]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_terminal_tasks(path)[0].asset_hash, task.asset_hash)
            (task.task_dir / "tests" / "test.sh").write_text("changed verifier", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_terminal_tasks(path)
            manifest["tasks"][0]["path"] = "../outside"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_terminal_tasks(path)

    def test_boolean_restoration_claim_does_not_certify_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_assets(Path(tmp) / "task")
            backend = InjectedTerminalBackend()
            env = TerminalEnvironment(task, {"paired_scope": PROTOCOL}, backend=backend)
            env.reset()
            self.assertFalse(env.capabilities()["restore_certified"])
            self.assertTrue(env.capabilities()["randomized_single_supported"])
            self.assertFalse(env.configuration()["official_harbor_session_equivalence"])

    def test_shell_failure_continues_and_verifier_is_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_assets(Path(tmp) / "task")
            backend = InjectedTerminalBackend()
            env = TerminalEnvironment(task, backend=backend)
            initial = env.reset()
            self.assertNotIn("PRIVATE-", canonical(initial))
            first = env.step({"type": "shell", "command": "exit 7"})
            self.assertTrue(first.valid)
            self.assertFalse(first.terminal)
            self.assertIsNone(first.success)
            self.assertEqual(backend.verifications, [])
            checkpoint = env.snapshot()
            restored = TerminalEnvironment(task, backend=InjectedTerminalBackend())
            restored.restore(checkpoint)
            self.assertEqual(restored.state_fingerprint(), env.state_fingerprint())
            final = env.step({"type": "finish"})
            self.assertTrue(final.terminal)
            self.assertIs(final.success, False)
            self.assertNotIn("PRIVATE-", canonical(final.context))
            self.assertNotIn("private_verifier_log", canonical(final.context))
            self.assertEqual(len(backend.verifications), 1)

    def test_verifier_infrastructure_error_never_becomes_failure_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_assets(Path(tmp) / "task")
            env = TerminalEnvironment(task, backend=InjectedTerminalBackend(verify_error=True))
            env.reset()
            record = InterventionCollector(StudySpec(max_steps=3, max_model_calls=3)).collect_deployment(
                env, ShellThenFinishActor(), None, NeverCalledPolicy(), trial_id="verifier-error")
            self.assertFalse(record["valid"])
            self.assertIsNone(record["arms"]["silent"]["success"])
            self.assertIsNone(record["arms"]["silent"]["utility"])
            self.assertEqual(record["arms"]["silent"]["status"], "error")
            self.assertIsNone(record["paired_regression"])

    def test_no_memory_counts_only_real_gate_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_assets(Path(tmp) / "task")
            env = TerminalEnvironment(task, backend=InjectedTerminalBackend())
            env.reset()
            record = InterventionCollector(StudySpec(max_steps=3, max_model_calls=3)).collect_deployment(
                env, ShellThenFinishActor(), None, NeverCalledPolicy(), trial_id="no-memory-gate-count")
            self.assertTrue(record["valid"], record.get("error"))
            self.assertEqual(record["gate_costs"].get("gate_calls", 0), 0)

    def test_prefix_is_frozen_once_before_deployment_replicates(self):
        from cave_mem_v2.terminal_workflow import rollout_terminal

        class UniqueImageBackend(InjectedTerminalBackend):
            def execute(self, image, command, protocol, *, seed=0):
                _, result = super().execute(image, command, protocol, seed=seed)
                return "sha256:" + digest([len(self.executions), command]), result

        class PrefixActor(ShellThenFinishActor):
            def propose(self, context, *, stage, seed, reminder=None, call_id=""):
                if stage == "prefix":
                    action = {"type": "shell", "command": "prefix"}
                    return ActorResponse(action, canonical(action), {"model_calls": 1, "physical_attempts": 0,
                        "input_tokens": 0, "output_tokens": 0, "api_cost": 0}, digest(context), (), (), "prefix-actor")
                return super().propose(context, stage=stage, seed=seed, reminder=reminder, call_id=call_id)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = {}
            for name in ("target-a", "target-b"):
                task = task_assets(root / name)
                (task.task_dir / "instruction.md").write_text("Create the artifact for " + name, encoding="utf-8")
                tasks[name] = task_from_directory(task.task_dir, repository_commit=COMMIT, protocol={"max_steps": 3})
            backend, bank = UniqueImageBackend(), MemoryBank()
            spec = StudySpec(max_steps=2, max_model_calls=3, replicates=2)
            raw = {"terminal": {"checkpoint_steps": 1}}
            identity = {"config_hash": "test-config", "manifest_hash": "test-manifest"}
            rows = [{"trial_id": name, "sampling_weight": 1.} for name in tasks]
            split = {"manifest_hash": "split-hash", "trial_assignments": {
                name: {"group_id": name + "-group", "split": "test"} for name in tasks}}
            cohort = {**identity, "bank_hash": bank.content_hash(), "split_hash": split["manifest_hash"],
                      "source_disjoint": False, "task_map": {name: name for name in tasks}}
            cohort["checksum"] = digest(cohort)
            write_json(root / "collection" / "terminal-collection.json", cohort)
            write_json(root / "dataset" / "split-manifest.json", split)
            write_json(root / "models" / "training.json", {"dataset_manifest": split["manifest_hash"]})
            # Exercise orchestration/runtime with real terminal state objects; only prior-artifact
            # setup is injected here. The separate workflow suite tests artifact production.
            with patch("cave_mem_v2.terminal_workflow._setup", return_value=(spec, raw, tasks,
                       PrefixActor, lambda task: TerminalEnvironment(task, backend=backend))), \
                 patch("cave_mem_v2.terminal_workflow._identity", return_value=identity), \
                 patch("cave_mem_v2.terminal_workflow._bank", return_value=bank), \
                 patch("cave_mem_v2.terminal_workflow.load_partitions", return_value={"test": rows}):
                report = rollout_terminal("config", "manifest", "sources", root / "collection", root / "dataset",
                                          root / "models", root / "rollout", policy_file=None)
            self.assertEqual(report["completed"], 4)
            self.assertEqual(sum(command == "prefix" for _, command, _ in backend.executions), 2)
            records = [read_json(path) for path in (root / "rollout" / "annotated").glob("*.json")]
            for name in tasks:
                group = [r for r in records if r["trial"]["trial_id"] == name]
                self.assertEqual(len(group), 2)
                self.assertEqual(len({r["trial"]["checkpoint_id"] for r in group}), 1)
            ledger = read_jsonl(root / "rollout" / "prefix-costs.jsonl")
            self.assertEqual(sum(row["physical_prefix_charged"] for row in ledger), 2)


if __name__ == "__main__":
    unittest.main()
