"""Measure filesystem restore mechanics; never approve task scope automatically."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from cave_mem_v2.contracts import write_json
from cave_mem_v2.terminal import TerminalEnvironment, load_terminal_tasks


def diagnose(task, *, backend=None):
    env = TerminalEnvironment(task, backend=backend)
    try:
        env.reset()
        base = env.snapshot()
        report = env.diagnose_restoration()
        for row in report["restoration_checks"]:
            row["check_id"] = "initial-" + row["check_id"]
        # Write only inside a disposable image branch, outside task paths.
        marker = "/tmp/cave-restore-probe-" + uuid.uuid4().hex
        token = uuid.uuid4().hex
        if task.spec["protocol"]["max_steps"] < 3:
            raise ValueError("restore probe requires at least three task steps")
        env.step({"type": "shell", "command": f"printf '%s' '{token}' > '{marker}'"})
        changed = env.snapshot()
        changed_report = env.diagnose_restoration()
        for row in changed_report["restoration_checks"]:
            row["check_id"] = "after-write-" + row["check_id"]
        report["restoration_checks"].extend(changed_report["restoration_checks"])
        env.restore(changed)
        persisted = env.step({"type": "shell", "command": f"cat '{marker}'"})
        persisted_result = json.loads(persisted.context.observation)
        env.restore(base)
        isolated = env.step({"type": "shell", "command": f"test ! -e '{marker}'"})
        isolated_result = json.loads(isolated.context.observation)
        if persisted_result["exit_code"] != 0 or persisted_result["stdout"] != token:
            raise ValueError("restored branch did not preserve its filesystem write")
        if isolated_result["exit_code"] != 0:
            raise ValueError("filesystem write leaked into the immutable base checkpoint")
        report["filesystem_probe"] = {"persisted_after_restore": True, "base_branch_isolated": True,
                                       "changed_image": changed["state"]["image"]}
        report["status"] = "diagnostics_passed_but_task_scope_unreviewed"
        # Reviewer and task assumptions deliberately remain unapproved.
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", help="omit to check all manifest tasks")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("diagnostic output directory must be empty")
    tasks = load_terminal_tasks(args.manifest)
    if args.task_id:
        tasks = [t for t in tasks if t.task_id == args.task_id]
        if not tasks:
            parser.error("task ID is not in the manifest")
    failures = []
    for task in tasks:
        try:
            report = diagnose(task)
            write_json(output / (task.task_id + ".json"), report)
        except Exception as exc:
            failure = {"task_id": task.task_id, "error": type(exc).__name__, "message": str(exc)}
            failures.append(failure)
            write_json(output / (task.task_id + ".error.json"), failure)
    print(json.dumps({"checked": len(tasks), "failed": len(failures),
                      "paired_certified": False, "output": str(output.resolve())}, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
