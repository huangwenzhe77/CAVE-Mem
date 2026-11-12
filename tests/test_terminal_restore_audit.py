"""Restore diagnostics must test file contents and branch isolation, not only hashes."""
import copy
import importlib.util
import shlex
import tempfile
import unittest
from pathlib import Path

from cave_mem_v2.contracts import digest
from test_terminal_audit import task_assets


script_path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_terminal_restore.py"
module_spec = importlib.util.spec_from_file_location("terminal_restore_diagnostic", script_path)
restore_diagnostic = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(restore_diagnostic)


class FilesystemProbeBackend:
    identity = "injected-filesystem-probe.v1"
    filesystem_boundaries = True

    def __init__(self, *, lose_writes=False, leak_base=False):
        self.base = "sha256:" + "a" * 64
        self.images = {self.base: {}}
        self.lose_writes, self.leak_base = lose_writes, leak_base
        self.commands = []

    def ensure_image(self, task, protocol):
        return self.base

    def image_available(self, image):
        return image in self.images

    def execute(self, image, command, protocol, *, seed=0):
        self.commands.append(command)
        files = copy.deepcopy(self.images[image])
        parts = shlex.split(command)
        exit_code, stdout = 0, ""
        if parts[0] == "printf":
            if not self.lose_writes:
                files[parts[-1]] = parts[2]
            if self.leak_base:
                self.images[self.base] = copy.deepcopy(files)
        elif parts[0] == "cat":
            if parts[1] not in files:
                exit_code = 1
            else:
                stdout = files[parts[1]]
        elif parts[:3] == ["test", "!", "-e"]:
            exit_code = int(parts[3] in files)
        else:
            raise AssertionError("diagnostic issued an unreviewed command")
        next_image = "sha256:" + digest([len(self.commands), files])
        self.images[next_image] = files
        return next_image, {"exit_code": exit_code, "stdout": stdout, "stderr": "", "truncated": False}

    def verify(self, image, task, protocol):
        raise AssertionError("restore diagnostic must not invoke a task verifier")


class TerminalRestoreAuditTests(unittest.TestCase):
    def test_matching_serialization_is_not_automatic_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_assets(Path(tmp) / "task")
            backend = FilesystemProbeBackend()
            report = restore_diagnostic.diagnose(task, backend=backend)
            self.assertEqual(report["status"], "diagnostics_passed_but_task_scope_unreviewed")
            self.assertTrue(report["filesystem_probe"]["persisted_after_restore"])
            self.assertTrue(report["filesystem_probe"]["base_branch_isolated"])
            self.assertEqual(report["reviewer"], "")
            self.assertFalse(any(report["reviewed_constraints"].values()))
            self.assertEqual(len(report["restoration_checks"]), 4)
            self.assertEqual(len(backend.commands), 3)

    def test_matching_snapshot_hash_cannot_hide_lost_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_assets(Path(tmp) / "task")
            with self.assertRaisesRegex(ValueError, "preserve"):
                restore_diagnostic.diagnose(task, backend=FilesystemProbeBackend(lose_writes=True))

    def test_matching_snapshot_hash_cannot_hide_cross_branch_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_assets(Path(tmp) / "task")
            with self.assertRaisesRegex(ValueError, "leaked"):
                restore_diagnostic.diagnose(task, backend=FilesystemProbeBackend(leak_base=True))


if __name__ == "__main__":
    unittest.main()
