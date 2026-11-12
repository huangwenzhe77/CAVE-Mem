"""Explicit local commands. Development defaults never contact a model API."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cave2", description="proposal-2 v/rho/e implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="report interpreter and optional dependencies")
    for name in ("terminal-preflight", "terminal-pilot", "terminal-source", "terminal-collect",
                 "terminal-select", "terminal-rollout", "terminal-pipeline"):
        p = commands.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--config", required=name != "terminal-preflight")
        if name != "terminal-preflight":
            p.add_argument("--real-actor", action="store_true")
        if name == "terminal-pilot":
            p.add_argument("--role", choices=("source", "target"), default="source")
        if name == "terminal-source":
            p.add_argument("--pilot", required=True)
            p.add_argument("--candidates", required=True)
        if name in ("terminal-collect", "terminal-select", "terminal-rollout", "terminal-pipeline"):
            p.add_argument("--sources", required=True)
        if name in ("terminal-collect", "terminal-pipeline"):
            p.add_argument("--kind", choices=("paired", "single"), default="paired")
        if name in ("terminal-select", "terminal-rollout"):
            for option in ("collection", "dataset", "models"):
                p.add_argument("--" + option, required=True)
        if name == "terminal-rollout":
            p.add_argument("--split", choices=("selection", "audit", "test"), default="test")
            system = p.add_mutually_exclusive_group()
            system.add_argument("--no-memory", action="store_true")
            system.add_argument("--baseline", choices=("silent", "always_remind", "source_only", "applicability", "exposure", "hard_vre", "relevance"))
            p.add_argument("--relevance-threshold", type=float, default=0.)
    for name in ("source", "collect", "dataset", "train", "select", "evaluate", "rollout", "pipeline"):
        p = commands.add_parser(name)
        if name not in ("dataset", "evaluate"):
            p.add_argument("--config", default="configs/development.toml")
        p.add_argument("--output", required=True)
        if name in ("collect", "rollout", "select"):
            p.add_argument("--sources", required=True)
        if name == "dataset":
            p.add_argument("--collection", required=True)
        if name in ("train", "evaluate", "rollout", "select"):
            p.add_argument("--dataset", required=True)
        if name in ("evaluate", "rollout", "select"):
            p.add_argument("--models", required=True)
            if name != "select":
                p.add_argument("--split", choices=("audit", "test") if name == "evaluate" else ("selection", "audit", "test"), default="test")
        if name in ("collect", "pipeline"):
            p.add_argument("--kind", choices=("paired", "single"), default="paired")
        if name in ("source", "collect", "rollout", "select", "pipeline"):
            p.add_argument("--real-actor", action="store_true", help="explicitly use the configured model API")
        if name == "rollout":
            p.add_argument("--no-memory", action="store_true", help="run the complete actor without retrieval/reminders")
    args = parser.parse_args(argv)
    try:
        from . import workflow as w
        if args.command.startswith("terminal-"):
            from . import terminal_workflow as terminal
            if args.command == "terminal-preflight":
                result = terminal.preflight_terminal(args.manifest, args.output, config=args.config)
            elif args.command == "terminal-pilot":
                result = terminal.pilot_terminal(args.config, args.manifest, args.output,
                                                role=args.role, real_actor=args.real_actor)
            elif args.command == "terminal-source":
                result = terminal.build_terminal_sources(args.config, args.manifest, args.pilot,
                    args.candidates, args.output, real_actor=args.real_actor)
            elif args.command == "terminal-collect":
                result = terminal.collect_terminal(args.config, args.manifest, args.sources, args.output,
                    kind=args.kind, real_actor=args.real_actor)
            elif args.command == "terminal-select":
                result = terminal.select_terminal(args.config, args.manifest, args.sources, args.collection,
                    args.dataset, args.models, args.output, real_actor=args.real_actor)
            elif args.command == "terminal-rollout":
                result = terminal.rollout_terminal(args.config, args.manifest, args.sources, args.collection,
                    args.dataset, args.models, args.output, split=args.split,
                    policy_file=None if args.no_memory else "policy.json", baseline=args.baseline,
                    relevance_threshold=args.relevance_threshold, real_actor=args.real_actor)
            else:
                result = terminal.run_terminal_pipeline(args.config, args.manifest, args.sources,
                    args.output, kind=args.kind, real_actor=args.real_actor)
        elif args.command == "doctor":
            result = w.provenance()
        elif args.command == "source":
            r = w.build_development_sources(args.config, args.output, real_actor=args.real_actor)
            result = {"source_groups": r["source_groups"], "target_cases": len(r["cases"]), "suite_hash": r["suite_hash"]}
        elif args.command == "collect":
            result = w.collect_development(args.config, args.sources, args.output, kind=args.kind, real_actor=args.real_actor)
        elif args.command == "dataset":
            result = w.export_dataset(args.collection, args.output)
        elif args.command == "train":
            r = w.train_models(args.config, args.dataset, args.output)
            result = {"kind": r["kind"], "method": r["policy_report"]["method"]}
        elif args.command == "evaluate":
            r = w.evaluate_models(args.dataset, args.models, args.output, split=args.split)
            result = {"kind": r["kind"], "split": r["split"], "policies": list(r["policies"])}
        elif args.command == "rollout":
            result = w.live_evaluate(args.config, args.sources, args.dataset, args.models, args.output,
                                     split=args.split, real_actor=args.real_actor,
                                     policy_file=None if args.no_memory else "policy.json")
        elif args.command == "select":
            r = w.select_with_rollouts(args.config, args.sources, args.dataset, args.models, args.output,
                                       real_actor=args.real_actor)
            result = {"winner": r["winner"], "fallback_silent": r["fallback_silent"]}
        else:
            result = w.run_pipeline(args.config, args.output, kind=args.kind, real_actor=args.real_actor)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"cave2: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
