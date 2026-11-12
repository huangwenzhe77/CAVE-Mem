"""Content-versioned memories and frozen single-candidate retrieval."""
from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import fields
from pathlib import Path
from typing import Iterable

from .contracts import PublicContext, SourceMemory, canonical, digest, finite, plain, read_jsonl


def memory_from_dict(value: dict) -> SourceMemory:
    allowed = {f.name for f in fields(SourceMemory)}
    if set(value) - allowed:
        raise ValueError("unknown memory fields")
    return SourceMemory(**value)


class MemoryBank:
    """Append-only immutable records; same ID/different content is an error."""
    def __init__(self, path: str | Path | None = None, memories: Iterable[SourceMemory] = ()):
        self.path = Path(path) if path is not None else None
        self._memories: dict[str, dict] = {}
        if self.path and self.path.exists():
            for row in read_jsonl(self.path):
                if set(row) != {"memory", "content_hash"}:
                    raise ValueError("unsupported bank record")
                memory = memory_from_dict(row["memory"])
                if row["content_hash"] != memory.content_hash():
                    raise ValueError("bank record checksum mismatch")
                self._insert(memory)
        for memory in memories:
            self.put(memory)

    def _insert(self, memory: SourceMemory) -> bool:
        if not memory.memory_id:
            raise ValueError("memory ID required")
        row = plain(memory)
        existing = self._memories.get(memory.memory_id)
        if existing is not None:
            if canonical(existing) != canonical(row):
                raise ValueError(f"immutable memory ID conflict: {memory.memory_id}")
            return False
        self._memories[memory.memory_id] = row
        return True

    def put(self, memory: SourceMemory) -> None:
        if self._insert(memory) and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical({"memory": plain(memory), "content_hash": memory.content_hash()}) + "\n")

    def list_all(self) -> list[SourceMemory]:
        return [memory_from_dict(copy.deepcopy(self._memories[k])) for k in sorted(self._memories)]

    def get(self, memory_id: str) -> SourceMemory | None:
        row = self._memories.get(memory_id)
        return None if row is None else memory_from_dict(copy.deepcopy(row))

    def content_hash(self) -> str:
        return digest([plain(m) for m in self.list_all()])


def memory_text(memory: SourceMemory) -> str:
    # IDs/provenance and outcome labels do not rank retrieved memories.
    return "\n".join((memory.mechanism, canonical(memory.problem_action), canonical(memory.correction),
                      canonical(memory.positive_conditions), canonical(memory.exclusion_conditions)))


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold()))


class FixedRetriever:
    """A frozen top-1 lexical or embedding retriever; no outcome/Q updates."""
    def __init__(self, memories: MemoryBank | Iterable[SourceMemory], *, encoder=None,
                 min_score: float | None = None, history_window: int = 4):
        self._memories = tuple(copy.deepcopy(memories.list_all() if isinstance(memories, MemoryBank) else list(memories)))
        if len({m.memory_id for m in self._memories}) != len(self._memories):
            raise ValueError("duplicate candidate IDs")
        self.encoder = encoder
        self.encoder_version = getattr(encoder, "version", None)
        if encoder is not None and not self.encoder_version:
            raise ValueError("retrieval encoder must expose a frozen version")
        self.min_score = None if min_score is None else finite(min_score, "retrieval threshold")
        if type(history_window) is not int or history_window < 0:
            raise ValueError("history_window must be a nonnegative integer")
        self.history_window = history_window
        self._query_config_hash = digest([self.min_score, history_window])
        self.bank_hash = digest([plain(m) for m in self._memories])
        self.version = digest({"retriever": "fixed-top1.v2", "bank": self.bank_hash,
                               "encoder": getattr(encoder, "version", None), "threshold": min_score,
                               "query": "public-instruction-observation-recent-executed-history.v1",
                               "history_window": history_window})
        self._embeddings = None if encoder is None else encoder.encode([memory_text(m) for m in self._memories])
        self.last_usage = {}

    def retrieve(self, context: PublicContext) -> tuple[SourceMemory | None, float]:
        started = time.perf_counter()
        self.last_usage = {"retrieval_calls": 1, "encoder_calls": 0, "wall_seconds": 0.0,
                           "api_cost": 0.0 if self.encoder is None else None}
        if not isinstance(context, PublicContext):
            raise TypeError("retrieval accepts only a public context")
        if getattr(self.encoder, "version", None) != self.encoder_version:
            raise ValueError("frozen retrieval encoder version changed")
        if digest([self.min_score, self.history_window]) != self._query_config_hash:
            raise ValueError("frozen retrieval threshold/history settings changed")
        if digest([plain(m) for m in self._memories]) != self.bank_hash:
            raise ValueError("frozen candidate bank changed")
        if not self._memories:
            return None, 0.0
        query = context.instruction + "\n" + context.observation
        if self.history_window:
            recent = context.history[-self.history_window:]
            if any(not isinstance(event, dict) or set(event) - {"action", "observation"} for event in recent):
                raise ValueError("retrieval history must contain executed public actions/observations only")
            from .features import _reject_private_fields
            for event in recent:
                _reject_private_fields(event.get("action"))
            query += "\n" + canonical(recent)
        if self.encoder is None:
            q = _tokens(query)
            scores = []
            for memory in self._memories:
                m = _tokens(memory_text(memory))
                scores.append(len(q & m) / math.sqrt(len(q) * len(m)) if q and m else 0.0)
        else:
            q = self.encoder.encode([query])[0]
            self.last_usage["encoder_calls"] = getattr(self.encoder, "last_usage", {}).get("encoder_calls", 1)
            scores = []
            for m in self._embeddings:
                if len(q) != len(m):
                    raise ValueError("encoder dimensions changed")
                denominator = math.sqrt(sum(v*v for v in q) * sum(v*v for v in m))
                scores.append(sum(a*b for a,b in zip(q,m)) / denominator if denominator else 0.0)
        index = min(range(len(scores)), key=lambda i: (-scores[i], self._memories[i].memory_id))
        score = finite(scores[index], "retrieval score")
        self.last_usage["wall_seconds"] = time.perf_counter() - started
        if self.min_score is not None and score < self.min_score:
            return None, score
        return copy.deepcopy(self._memories[index]), score
