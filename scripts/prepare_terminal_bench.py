"""Freeze a clean, pinned Terminal-Bench checkout and create unapproved audits.

This command neither downloads task solutions nor runs Docker/the actor. It
does not certify tasks. Use the separate restoration diagnostic and review the
task's execution assumptions before filling in an audit.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from cave_mem_v2.config import load_config
from cave_mem_v2.contracts import write_json
from cave_mem_v2.terminal import PROTOCOL, protocol_hash, task_from_directory


PINNED_COMMIT = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"


def prepare(repository, config, output, *, commit=PINNED_COMMIT):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("use a full immutable Git commit")
    actual = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != commit:
        raise ValueError(f"checkout must be at {commit}; found {actual}")
    dirty = subprocess.run(["git", "-C", str(repository), "status", "--porcelain",
                            "--untracked-files=normal"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("benchmark checkout has local changes; use a clean dedicated checkout")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("manifest output directory must be empty")
    if output.is_relative_to(repository):
        raise ValueError("write manifests outside the pinned benchmark checkout")
    spec, raw = load_config(config)
    settings = raw.get("terminal", {})
    source, target = settings.get("source_ids", []), settings.get("target_ids", [])
    if not source or not target or set(source) & set(target):
        raise ValueError("declare separate nonempty source_ids and target_ids")
    ids = source + target
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be unique")
    if any(not isinstance(task_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", task_id)
           for task_id in ids):
        raise ValueError("task IDs must be top-level Terminal-Bench directory names")
    ignored = subprocess.run(["git", "-C", str(repository), "ls-files", "--others", "--ignored",
                              "--exclude-standard", "--", *ids], check=True,
                             capture_output=True, text=True).stdout.strip()
    if ignored:
        raise ValueError("selected task contains ignored local files outside the pinned commit")
    protocol = dict(settings.get("protocol", {}))
    protocol["max_steps"] = spec.max_steps + settings.get("checkpoint_steps", 0)
    protocol["paired_scope"] = PROTOCOL
    rows, audits = [], []
    for task_id in ids:
        task_path = (repository / task_id).resolve()
        if not task_path.is_relative_to(repository):
            raise ValueError("task escapes benchmark checkout")
        task = task_from_directory(task_path, task_id, repository_commit=commit, protocol=protocol)
        task_protocol = dict(task.spec["protocol"])
        task_protocol["audit_path"] = "audits/" + task_id + ".json"
        rows.append({"task_id": task_id, "path": task_id, "asset_hash": task.asset_hash,
                     "protocol": task_protocol})
        audits.append((task_protocol["audit_path"], {
            "schema_version": "terminal-task-audit.v1",
            "scope": PROTOCOL,
            "repository_commit": commit,
            "asset_hash": task.asset_hash,
            "protocol_hash": protocol_hash(task_protocol),
            "backend": "docker-cli-fs-boundary.v1",
            "base_image": None,
            "reviewer": "",
            "review_notes": "",
            "reviewed_constraints": {
                "no_external_state": False,
                "no_persistent_services": False,
                "filesystem_boundary_semantics": False,
                "clock_and_rng_are_exogenous": False,
            },
            "restoration_checks": [],
            "status": "unreviewed; diagnostics and task review required before paired collection",
        }))
    manifest = {"schema_version": "terminal-task-manifest.v1", "benchmark": "terminal-bench@2.0",
                "repository_url": "https://github.com/laude-institute/terminal-bench-2",
                "repository_commit": commit, "root": str(repository), "tasks": rows}
    for relative, audit in audits:
        write_json(output / relative, audit)
    write_json(output / "manifest.json", manifest)
    return {"manifest": str(output / "manifest.json"), "tasks": len(rows),
            "source_tasks": len(source), "target_tasks": len(target),
            "paired_certified": False, "actor_called": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="clean checkout at the pinned commit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.repository, args.config, args.output, commit=args.commit), indent=2))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"prepare_terminal_bench: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
