"""Optional real ALFWorld randomized single-arm experiment.

Run --help without importing ALFWorld or contacting a provider. A model request
requires --allow-api. This example never enables certified paired collection.
Use a v2 ALFWorld source bank revalidated under its declared evidence scope;
legacy CAVE-Mem banks and the factorial development bank are not compatible
ALFWorld evidence merely because their records can be parsed.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from cave_mem_v2.alfworld import AlfworldEnvironment
from cave_mem_v2.config import load_config
from cave_mem_v2.contracts import digest, plain, stage_seed, write_json
from cave_mem_v2.features import FeatureBuilder, SentenceTransformerEncoder
from cave_mem_v2.memory import FixedRetriever, MemoryBank
from cave_mem_v2.providers import OpenAIConfig, OpenAICompatibleProvider, ProviderActor, parse_alfworld_action
from cave_mem_v2.runtime import InterventionCollector
from cave_mem_v2.source import validate_evidence


class VersionedSelector:
    def __init__(self, retriever):
        self.retriever, self.version = retriever, retriever.version
        self.encoder = retriever.encoder

    def __call__(self, context):
        memory, score = self.retriever.retrieve(context)
        usage = {"retrieval_calls": 1, "api_cost": 0.0}
        if self.encoder is not None:
            usage.update(self.encoder.last_usage)
            # Encoder API/energy billing is not inferred from a call count.
            usage["api_cost"] = None
        return memory, score, usage


class VersionedFeatures:
    def __init__(self, builder):
        self.builder, self.version = builder, builder.version

    def __call__(self, context, action, memory, checks, spec, *, retrieval_score=0.0):
        feature = self.builder.build(context, action, memory, checks,
                                     remaining_calls=spec.max_model_calls,
                                     retrieval_score=retrieval_score)
        return feature, self.builder.last_usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/provider.example.toml")
    parser.add_argument("--encoder-config", help="optional actual pinned encoder configuration")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--memory-store", required=True)
    parser.add_argument("--split", choices=("train", "eval_in_distribution", "eval_out_of_distribution"), default="eval_in_distribution")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--gamefile", help="optional exact asset path instead of relying on task ordering")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restore-repeats", type=int, default=3)
    parser.add_argument("--diagnose-only", action="store_true", help="run observable restoration checks, never call the model")
    parser.add_argument("--allow-api", action="store_true", help="explicitly allow the configured real model request(s)")
    args = parser.parse_args(argv)
    if not args.diagnose_only and not args.allow_api:
        parser.error("choose --diagnose-only or explicitly provide --allow-api")
    spec, raw = load_config(args.config)
    if not args.diagnose_only and not raw.get("provider"):
        parser.error("[provider] configuration is required for --allow-api")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bank = MemoryBank(args.memory_store)
    memories = bank.list_all()
    reports = [{"memory_id": m.memory_id, **validate_evidence(m)} for m in memories]
    if not memories or any(r["v"] != 1 for r in reports):
        raise ValueError("use a nonempty v2 source bank with supported replay evidence")
    if any(m.problem_action.get("type") != "alfworld_text" or m.correction.get("type") != "alfworld_text" for m in memories):
        raise ValueError("ALFWorld examples require source memories with actual ALFWorld action contracts")
    env = AlfworldEnvironment(split=args.split, task_index=args.task_index, gamefile=args.gamefile,
                             data_dir=args.data_dir, max_steps=spec.max_steps)
    try:
        env.reset(stage_seed(spec.base_seed, args.trial_id, 0, "reset"))
        checkpoint = env.snapshot()
        diagnostic = env.restore_report(checkpoint, args.restore_repeats)
        write_json(output / "restore-report.json", diagnostic)
        write_json(output / "source-bank-validation.json", reports)
        if env.capabilities()["restore_certified"]:
            raise RuntimeError("this example must not silently certify latent restoration")
        if args.diagnose_only:
            print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
            return 0
        if not diagnostic["observable_replays_matched"]:
            raise ValueError("observable restoration mismatched; inspect restore-report before a fresh single-arm run")
        env.restore(checkpoint)
        encoder = None
        if args.encoder_config:
            enc = tomllib.loads(Path(args.encoder_config).read_text(encoding="utf-8"))
            if set(enc) != {"encoder"}:
                raise ValueError("encoder config must contain only [encoder]")
            encoder = SentenceTransformerEncoder(cache_dir=output / "encoder-cache", **enc["encoder"])
        selector = VersionedSelector(FixedRetriever(memories, encoder=encoder))
        builder = VersionedFeatures(FeatureBuilder(encoder=encoder))
        actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig(**raw["provider"])),
                              parser=parse_alfworld_action,
                              system_prompt="You are an ALFWorld agent. Return exactly one executable action from the admissible action list. No thought, explanation, or multiple lines.",
                              prompt_version="alfworld-explicit-public-history.v2")
        collector = InterventionCollector(spec, artifact_dir=output / "records", feature_builder=builder)
        # Add explicit template/family keys in a preregistered batch study.
        # A benchmark-wide union key would incorrectly bind every task into
        # one dependency component and is not a substitute for semantic splits.
        record = collector.collect_single(env, actor, selector, trial_id=args.trial_id)
        write_json(output / "run.json", {"schema_version": "alfworld-single-example.v2",
                   "record_id": record["record_id"], "valid": record["valid"],
                   "spec_hash": digest(spec), "bank_hash": bank.content_hash(),
                   "source_evidence_scope": [m.evidence.get("scope") for m in memories],
                   "capabilities": env.capabilities(), "paired_harm": None,
                   "cost_note": "missing usage or money prices stay unknown; no paired risk claim"})
        print(json.dumps({"record_id": record["record_id"], "valid": record["valid"],
                          "observed": record.get("observed"), "error": record.get("error")}, ensure_ascii=False, indent=2))
        return 0 if record["valid"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
