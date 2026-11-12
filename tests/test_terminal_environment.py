from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from cave_mem_v2.contracts import PublicContext, SourceMemory, digest
from cave_mem_v2.features import MechanismAssessor, ObservationFactExtractor, assess
from cave_mem_v2.memory import FixedRetriever
from cave_mem_v2.terminal import (PROTOCOL, TerminalEnvironment, TerminalInfrastructureError,
    task_from_directory, load_terminal_tasks, parse_verifier_result, protocol_hash)


class FakeDocker:
    identity = "fake-docker-fs-test.v1"
    filesystem_boundaries = True
    def __init__(self):
        self.images = {"sha256:" + "a" * 64}
        self.commands, self.verifications = [], []
        self.fail = False
    def ensure_image(self, task, protocol):
        return "sha256:" + "a" * 64
    def image_available(self, image):
        return image in self.images
    def execute(self, image, command, protocol, seed=0):
        self.commands.append((image, command, protocol))
        if self.fail:
            raise TerminalInfrastructureError("daemon failed")
        output = "sha256:" + digest([image, command])
        self.images.add(output)
        return output, {"exit_code": 1 if command == "false" else 0, "stdout": "observed output", "stderr": ""}
    def verify(self, image, task, protocol):
        self.verifications.append((image, task.asset_hash))
        return {"success": True, "rewards": {"reward": 1}, "private_verifier_output": "SECRET TEST"}


def make_task(root):
    path = Path(root) / "task-one"
    (path / "environment").mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "solution").mkdir()
    (path / "instruction.md").write_text("Make the public artifact", encoding="utf-8")
    (path / "task.toml").write_text('version = "1.0"\n[environment]\n', encoding="utf-8")
    (path / "environment/Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n", encoding="utf-8")
    (path / "tests/test.sh").write_text("echo PRIVATE_TEST", encoding="utf-8")
    (path / "solution/solve.sh").write_text("echo PRIVATE_SOLUTION", encoding="utf-8")
    return task_from_directory(path, repository_commit="b" * 40)


class TerminalEnvironmentTests(unittest.TestCase):
    def test_manifest_freezes_all_assets_and_contains_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            task = make_task(temp)
            manifest = {"schema_version": "terminal-task-manifest.v1", "benchmark": "terminal-bench@2.0",
                "repository_commit": "b" * 40, "root": temp,
                "tasks": [{"task_id": task.task_id, "path": "task-one", "asset_hash": task.asset_hash}]}
            path = Path(temp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_terminal_tasks(path)[0].asset_hash, task.asset_hash)
            (task.task_dir / "tests/test.sh").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_terminal_tasks(path)
            manifest["tasks"][0]["path"] = "../outside"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_terminal_tasks(path)

    def test_shell_failure_continues_and_verifier_is_private(self):
        with tempfile.TemporaryDirectory() as temp:
            task, backend = make_task(temp), FakeDocker()
            env = TerminalEnvironment(task, backend=backend)
            before = env.reset()
            self.assertNotIn("PRIVATE", str(before))
            result = env.step({"type": "shell", "command": "false"})
            self.assertFalse(result.terminal)
            self.assertTrue(result.valid)
            self.assertIsNone(result.success)
            self.assertEqual(len(backend.verifications), 0)
            result = env.step({"type": "finish"})
            self.assertTrue(result.success)
            self.assertNotIn("SECRET TEST", str(result.context))
            self.assertEqual(env.verifier_report["rewards"], {"reward": 1})

    def test_restore_and_plain_flag_do_not_certify_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            task, backend = make_task(temp), FakeDocker()
            env = TerminalEnvironment(task, {"paired_scope": PROTOCOL}, backend=backend)
            env.reset()
            checkpoint = env.snapshot()
            env.step({"type": "shell", "command": "touch /app/x"})
            env.restore(checkpoint)
            self.assertEqual(env.state_fingerprint(), digest(checkpoint))
            self.assertFalse(env.capabilities()["restore_certified"])
            diagnostic = env.diagnose_restoration()
            self.assertFalse(any(diagnostic["reviewed_constraints"].values()))
            self.assertEqual(diagnostic["restoration_checks"][0]["snapshot_hash"],
                             diagnostic["restoration_checks"][0]["restored_hash"])

    def test_infrastructure_exception_remains_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            task, backend = make_task(temp), FakeDocker()
            env = TerminalEnvironment(task, backend=backend)
            env.reset()
            backend.fail = True
            with self.assertRaises(TerminalInfrastructureError):
                env.step({"type": "shell", "command": "anything"})
            self.assertEqual(env.terminal_status(), (False, None))

    def test_reward_zero_needs_actual_assertion_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reward.txt").write_text("0", encoding="utf-8")
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)
            report = {"results": {"summary": {"tests": 1, "passed": 0, "failed": 1, "skipped": 0, "pending": 0, "other": 0},
                                 "tests": [{"name": "artifact", "status": "failed", "message": "assert artifact == expected"}]}}
            (root / "ctrf.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(parse_verifier_result(root)["success"])
            report["results"]["tests"][0]["message"] = "ModuleNotFoundError: missing dependency"
            (root / "ctrf.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)
            (root / "reward.txt").write_text("nan", encoding="utf-8")
            with self.assertRaises(TerminalInfrastructureError):
                parse_verifier_result(root)

    def test_state_guard_changes_exposure_without_changing_draft(self):
        memory = SourceMemory("memory", "source", "trace", "checkpoint", "overwrite_without_backup",
            {"type": "shell", "command": "overwrite data"}, {"type": "shell", "command": "backup data"})
        assessor = MechanismAssessor({memory.mechanism: {
            "trigger_actions": [{"type": "shell", "command_regex": "overwrite .*"}],
            "safe_actions": [{"type": "shell", "command_regex": "backup .*"}],
            "protected_when": [{"field": "backed_up", "equals": True}]}})
        action = {"type": "shell", "command": "overwrite data"}
        make = lambda facts: PublicContext("task", "modify data", "public", facts)
        self.assertEqual(assessor(make({"backed_up": False}), action, memory)["e"], 1)
        self.assertEqual(assessor(make({"backed_up": True}), action, memory)["e"], 0)
        self.assertEqual(assessor(make({}), action, memory)["e"], "?")
        self.assertEqual(assess(make({}), action, memory)["e"], "?")
        self.assertEqual(assessor(make({}), {"type": "shell", "command": "wrapper"}, memory)["e"], "?")

    def test_retrieval_uses_public_history_and_can_return_no_candidate(self):
        memories = [SourceMemory(name, "s" + name, "trace" + name, name, name,
                                 {"type": name}, {"type": name}) for name in ("backup", "dependency")]
        context = PublicContext("task", "maintenance", "ready", history=(
            {"action": {"type": "backup"}, "observation": "backup backup"},))
        retriever = FixedRetriever(memories, min_score=.1)
        self.assertEqual(retriever.retrieve(context)[0].memory_id, "backup")
        unrelated = PublicContext("task", "abc", "xyz")
        self.assertIsNone(retriever.retrieve(unrelated)[0])
        self.assertEqual(retriever.last_usage["retrieval_calls"], 1)
        private = PublicContext("task", "public", "public", history=(
            {"action": {"type": "backup", "expert_action": "hidden"}, "observation": "public"},))
        with self.assertRaisesRegex(ValueError, "private"):
            retriever.retrieve(private)
        retriever.history_window = 0
        with self.assertRaisesRegex(ValueError, "settings changed"):
            retriever.retrieve(context)

    def test_public_fact_requires_diagnostic_provenance_and_expires(self):
        diagnostic = {"type": "shell", "command": "python /app/check_backup.py"}
        rule = {"source": "history", "diagnostic_actions": [diagnostic],
                "diagnostic_basis": "Reviewed read-only checksum checker for this task asset revision.",
                "patterns": [{"regex": "backup verified", "value": True},
                             {"regex": "backup absent", "value": False}]}
        extractor = ObservationFactExtractor({"backed_up": rule})
        output = json.dumps({"exit_code": 0, "stdout": "backup verified", "stderr": ""})
        event = {"action": diagnostic, "observation": output}
        make = lambda history: PublicContext("task", "modify data", history[-1]["observation"] if history else "ready", history=tuple(history))
        self.assertEqual(extractor.extract(make([event])), {"backed_up": True})
        echoed = {"action": {"type": "shell", "command": "echo 'backup verified'"}, "observation": output}
        self.assertEqual(extractor.extract(make([echoed])), {})
        mutated = {"action": {"type": "shell", "command": "rm /app/backup"}, "observation": json.dumps({"exit_code": 0, "stdout": ""})}
        self.assertEqual(extractor.extract(make([event, mutated])), {})
        self.assertEqual(extractor.extract(make([])), {})
        for flag, value in (("exit_code", 1), ("truncated", True), ("command_timed_out", True)):
            invalid = json.dumps({"exit_code": 0, "stdout": "backup verified", flag: value})
            self.assertEqual(extractor.extract(make([{"action": diagnostic, "observation": invalid}])), {})
        with self.assertRaisesRegex(ValueError, "private"):
            ObservationFactExtractor({"reward": rule})
        with self.assertRaisesRegex(ValueError, "diagnostic_actions"):
            ObservationFactExtractor({"backed_up": {"source": "history", "patterns": rule["patterns"]}})

    def test_fact_persistence_only_across_reviewed_actions(self):
        diagnostic = {"type": "shell", "command": "check-backup"}
        harmless = {"type": "shell", "command": "pwd"}
        rule = {"source": "history", "diagnostic_actions": [diagnostic], "diagnostic_basis": "reviewed checker",
                "patterns": [{"regex": "backup verified", "value": True}],
                "persistence": "reviewed_actions", "preserve_actions": [harmless],
                "persistence_basis": "pwd cannot modify the registered backup resource"}
        extractor = ObservationFactExtractor({"backed_up": rule})
        output = json.dumps({"exit_code": 0, "stdout": "backup verified"})
        history = [{"action": diagnostic, "observation": output},
                   {"action": harmless, "observation": json.dumps({"exit_code": 0, "stdout": "/app"})}]
        context = PublicContext("task", "modify", "public", history=tuple(history))
        self.assertEqual(extractor.extract(context), {"backed_up": True})
        history.append({"action": {"type": "shell", "command": "unknown-mutator"}, "observation": ""})
        context = PublicContext("task", "modify", "public", history=tuple(history))
        self.assertEqual(extractor.extract(context), {})

    def test_fixture_guard_fixed_after_preparation(self):
        memory = SourceMemory("memory", "source", "trace", "checkpoint", "prepare before operate",
            {"type": "operate"}, {"type": "prepare"}, ({"field": "requires_preparation", "equals": True},))
        for prepared in (False, True):
            context = PublicContext("task", "modify", "public", {"requires_preparation": True, "prepared": prepared})
            result = assess(context, {"type": "operate"}, memory)
            self.assertEqual(result["rho"], 1)
            self.assertEqual(result["e"], int(not prepared))


if __name__ == "__main__":
    unittest.main()
