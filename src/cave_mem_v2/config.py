"""Strict local TOML configuration without implicit .env loading."""
from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

from .contracts import StudySpec, digest


def load_config(path: str | Path) -> tuple[StudySpec, dict]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"study", "experiment", "training", "provider", "environment", "encoder", "terminal", "source", "retrieval"}
    if set(raw) - allowed:
        raise ValueError(f"unknown configuration sections: {sorted(set(raw) - allowed)}")
    study = raw.get("study", {})
    valid = {f.name for f in dataclasses.fields(StudySpec)}
    if set(study) - valid:
        raise ValueError(f"unknown study fields: {sorted(set(study) - valid)}")
    spec = StudySpec(**study)
    return spec, raw


def configuration_fingerprint(spec: StudySpec, raw: dict) -> str:
    return digest({"resolved_study": dataclasses.asdict(spec), "configuration": raw})
