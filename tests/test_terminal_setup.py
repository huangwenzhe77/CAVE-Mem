"""Deployment helpers must freeze assets and never manufacture certification."""
import copy
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cave_mem_v2.contracts import digest
from cave_mem_v2.terminal import TerminalEnvironment, load_terminal_tasks, task_from_directory
from scripts.prepare_terminal_bench import prepare
from scripts.diagnose_terminal_restore import diagnose


def task_files(root, name):
    task = Path(root) / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Build public artifact " + name, encoding="utf-8")
    (task / "task.toml").write_text('schema_version = "1.1"\n', encoding="utf-8")
    (task / "environment/Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (task / "tests/test.sh").write_text("true\n", encoding="utf-8")
    return task


class FilesystemBackend:
    identity = "helper-test-filesystem.v1"
    filesystem_boundaries = True

    def __init__(self, leak=False):
        self.base = "sha256:" + "a" * 64
        self.images = {self.base: {}}
        self.leak = leak

    def ensure_image(self, task, protocol):
        return self.base

    def image_available(self, image):
        return image in self.images

    def execute(self, image, command, protocol, *, seed=0):
        files = self.images[image] if self.leak else copy.deepcopy(self.images[image])
        args = shlex.split(command)
        output, exit_code = "", 0
        if args[0] == "printf":
            files[args[-1]] = args[2]
        elif args[0] == "cat":
            output = files.get(args[-1], "")
            exit_code = int(args[-1] not in files)
        elif args[:3] == ["test", "!", "-e"]:
            exit_code = int(args[-1] in files)
        else:
            raise AssertionError("unexpected diagnostic command")
        next_image = "sha256:" + digest([files, command])
        self.images[next_image] = files
        return next_image, {"exit_code": exit_code, "stdout": output, "stderr": ""}


class TerminalSetupTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "git is required for checkout verification")
    def test_clean_commit_pinned_and_audits_unapproved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "benchmark"
            repo.mkdir()
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True, capture_output=True)
            task_files(repo, "source")
            task_files(repo, "target")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Setup test", "-c",
                            "user.email=setup@example.invalid", "commit", "--quiet", "-m", "fixture"],
                           check=True, capture_output=True)
            commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True).stdout.strip()
            config = root / "study.toml"
            config.write_text('[study]\nmax_steps=3\n[terminal]\nsource_ids=["source"]\n'
                              'target_ids=["target"]\ncheckpoint_steps=1\nsource_disjoint=false\n', encoding="utf-8")
            report = prepare(repo, config, root / "manifest", commit=commit)
            self.assertFalse(report["paired_certified"])
            tasks = load_terminal_tasks(report["manifest"])
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0].spec["protocol"]["max_steps"], 4)
            self.assertFalse(TerminalEnvironment(tasks[0]).capabilities()["restore_certified"])
            with self.assertRaisesRegex(ValueError, "checkout must"):
                prepare(repo, config, root / "other", commit="0" * 40)
            (repo / ".git/info/exclude").write_text("generated.bin\n", encoding="utf-8")
            generated = repo / "source/generated.bin"
            generated.write_bytes(b"not part of the official commit")
            with self.assertRaisesRegex(ValueError, "ignored local files"):
                prepare(repo, config, root / "ignored", commit=commit)
            generated.unlink()
            (repo / "target/instruction.md").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "local changes"):
                prepare(repo, config, root / "dirty", commit=commit)

    def test_diagnostic_checks_changed_image_and_keeps_review_unapproved(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_from_directory(task_files(tmp, "target"), repository_commit="a" * 40)
            report = diagnose(task, backend=FilesystemBackend())
            self.assertTrue(report["filesystem_probe"]["base_branch_isolated"])
            self.assertTrue(report["filesystem_probe"]["persisted_after_restore"])
            self.assertFalse(any(report["reviewed_constraints"].values()))
            self.assertEqual(report["reviewer"], "")
            self.assertEqual(len(report["restoration_checks"]), 4)
            self.assertEqual(len({c["check_id"] for c in report["restoration_checks"]}), 4)

    def test_diagnostic_rejects_cross_branch_file_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_from_directory(task_files(tmp, "target"), repository_commit="a" * 40)
            with self.assertRaisesRegex(ValueError, "leaked"):
                diagnose(task, backend=FilesystemBackend(leak=True))


if __name__ == "__main__":
    unittest.main()
