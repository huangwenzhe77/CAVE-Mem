"""Optional real ALFWorld reset/prefix adapter with conservative capabilities.

Imports ALFWorld only when opened. The adapter checks the selected game asset,
reset observations and every prefix result. That is observable replay evidence,
not an assertion that an opaque TextWorld latent state has been cloned exactly.
The default adapter therefore supports randomized single-arm experiments and
restore diagnostics, and refuses certified paired collection.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Callable

from .contracts import PublicContext, digest, plain
from .environments import EnvStep


def normalize_action(text: str) -> str:
    value = " ".join(str(text).strip().split()).lower()
    value = re.sub(r"^(?:action\s*\d*\s*:\s*|>\s*)", "", value).strip()
    match = re.match(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$", value)
    return f"move {match.group(1)} to {match.group(2)}" if match else value


def _batch(info: Any, key: str) -> list[str]:
    if not isinstance(info, dict):
        return []
    value = info.get(key, [])
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return [str(v) for v in value]
    return []


def _scalar(info: Any, key: str, default=None):
    value = info.get(key, default) if isinstance(info, dict) else default
    return value[0] if isinstance(value, (list, tuple)) and value else value


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _compatibility_patch() -> None:
    """Explicit versioned workaround for legacy TextWorld grammar evaluation."""
    from textworld.envs.pddl import textgen
    if getattr(textgen.EvalSymbol, "_cave_v2_explicit_context", False):
        return
    def derive(self, context=None):
        variables = dict((context or self.context)["variables"])
        return [textgen.TerminalSymbol(eval(self.expression, {}, variables))]
    textgen.EvalSymbol.derive = derive
    textgen.EvalSymbol._cave_v2_explicit_context = True


class AlfworldEnvironment:
    def __init__(self, *, split: str = "train", task_index: int = 0,
                 data_dir: str | Path | None = None, max_steps: int = 50,
                 gamefile: str | Path | None = None, expected_game_hash: str | None = None,
                 env_factory: Callable | None = None, compatibility_patch: bool = True) -> None:
        if split not in ("train", "eval_in_distribution", "eval_out_of_distribution"):
            raise ValueError("unsupported ALFWorld split")
        if type(task_index) is not int or task_index < 0 or type(max_steps) is not int or max_steps < 1:
            raise ValueError("invalid task_index or max_steps")
        self.split, self.task_index, self.max_steps = split, task_index, max_steps
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self.gamefile = str(Path(gamefile).resolve()) if gamefile else None
        self.expected_game_hash = expected_game_hash
        self.env_factory = env_factory
        self.use_compatibility_patch = compatibility_patch
        self._env = None
        self._raw_env = None
        self._prefix: list[str] = []
        self._observed_trace: list[dict] = []
        self._observation = ""
        self._admissible: tuple[str, ...] = ()
        self._terminal = False
        self._success = False
        self._instruction = "Complete the ALFWorld task."
        self._game_hash: str | None = None
        self._reset_seed: int | None = None

    def configuration(self) -> dict:
        return {"adapter": "alfworld-reset-prefix.v2", "split": self.split,
                "max_steps": self.max_steps, "alfworld_version": _version("alfworld"),
                "textworld_version": _version("textworld"),
                "compatibility_patch": "explicit_context.v2" if self.use_compatibility_patch else None,
                "domain_randomization": False, "goal_description_randomization": False}

    def capabilities(self) -> dict:
        return {"restore": "reset_and_observable_prefix_replay", "restore_certified": False,
                "state_fingerprint_scope": "task_asset_and_observable_prefix_only",
                "future_randomness": "opaque_textworld_not_certified", "step_seed_applied": False,
                "reset_seed_requested": True, "reset_seed_confirmed": False,
                "isolated_sessions": True, "paired_primary_eligible": False,
                "randomized_single_supported": True, "usage": "tool_calls_exact_money_unknown"}

    def _create(self):
        if self.env_factory is not None:
            result = self.env_factory()
            return result if isinstance(result, tuple) else (result, result)
        import alfworld
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        if self.use_compatibility_patch:
            _compatibility_patch()
        root = self.data_dir or Path(alfworld.ALFWORLD_DATA)
        config = {"env": {"goal_desc_human_anns_prob": 0.0, "regen_game_files": False,
                          "domain_randomization": False, "task_types": [1, 2, 3, 4, 5, 6],
                          "expert_timeout_steps": 150, "expert_type": "handcoded"},
                  "dataset": {"data_path": str(root / "json_2.1.1" / "train"),
                              "eval_id_data_path": str(root / "json_2.1.1" / "valid_seen"),
                              "eval_ood_data_path": str(root / "json_2.1.1" / "valid_unseen"),
                              "num_train_games": -1, "num_eval_games": -1},
                  "logic": {"domain": str(root / "logic" / "alfred.pddl"), "grammar": str(root / "logic" / "alfred.twl2")},
                  "general": {"use_cuda": False, "training_method": "dagger"},
                  "dagger": {"training": {"max_nb_steps_per_episode": self.max_steps}},
                  "controller": {"type": "oracle", "debug": False, "load_receps": True}}
        raw = AlfredTWEnv(config, train_eval=self.split)
        if self.gamefile is not None:
            matches = [p for p in raw.game_files if str(Path(p).resolve()) == self.gamefile]
            if not matches:
                raise ValueError("requested gamefile is absent from this ALFWorld split")
            selected = matches[0]
        else:
            if self.task_index >= len(raw.game_files):
                raise IndexError("ALFWorld task index out of range")
            selected = raw.game_files[self.task_index]
        raw.game_files = [selected]
        raw.num_games = 1
        return raw.init_env(batch_size=1), raw

    def _asset_hash(self) -> str:
        files = getattr(self._raw_env, "game_files", None)
        chosen = self.gamefile or (str(files[0]) if files else None)
        if chosen is None:
            raise ValueError("environment must expose an explicit game asset")
        path = Path(chosen).resolve()
        if not path.is_file():
            raise FileNotFoundError("selected ALFWorld game asset is not a readable file")
        self.gamefile = str(path)
        # The game file plus colocated task description binds index to content.
        assets = {path.name: digest(path.read_bytes().hex())}
        for filename in ("traj_data.json", "game.tw-pddl", "initial_state.pddl"):
            related = path.parent / filename
            if related.is_file() and related != path:
                assets[filename] = digest(related.read_bytes().hex())
        return digest(assets)

    def _trace_item(self) -> dict:
        return {"observation": self._observation, "admissible": list(self._admissible),
                "terminal": self._terminal, "success": self._success}

    def reset(self, seed: int = 0) -> PublicContext:
        self.close()
        self._env, self._raw_env = self._create()
        self._game_hash = self._asset_hash()
        if self.expected_game_hash is not None and self._game_hash != self.expected_game_hash:
            self.close()
            raise ValueError("ALFWorld game asset hash mismatch")
        self._reset_seed = seed
        # TextWorld seed() support is adapter-specific; record request without
        # claiming that it seeds all latent randomness or the remote actor.
        seed_fn = getattr(self._env, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
        observations, infos = self._env.reset()
        self._observation = str(observations[0])
        self._admissible = tuple(normalize_action(v) for v in _batch(infos, "admissible_commands"))
        self._terminal = False
        self._success = bool(_scalar(infos, "won", False))
        match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", self._observation, flags=re.IGNORECASE)
        self._instruction = match.group(1).strip() if match else "Complete the ALFWorld task."
        self._prefix, self._observed_trace = [], [self._trace_item()]
        return self.observe()

    def observe(self) -> PublicContext:
        if self._env is None:
            raise RuntimeError("ALFWorld environment is not open")
        history = tuple({"action": {"type": "alfworld_text", "text": action},
                         "observation": self._observed_trace[i + 1]["observation"]}
                        for i, action in enumerate(self._prefix))
        # Never expose expert_plan, latent PDDL predicates or checkpoint controls.
        facts = {"phase": "terminal" if self._terminal else "acting"}
        return PublicContext("alfworld:" + str(self._game_hash), self._instruction, self._observation,
                             facts, history, self._admissible, max(0, self.max_steps - len(self._prefix)))

    def snapshot(self) -> dict:
        if self._env is None:
            raise RuntimeError("ALFWorld environment is not open")
        return {"schema_version": "alfworld-prefix-checkpoint.v2", "split": self.split,
                "task_index": self.task_index, "gamefile": self.gamefile, "game_hash": self._game_hash,
                "max_steps": self.max_steps, "reset_seed": self._reset_seed,
                "prefix": copy.deepcopy(self._prefix), "observed_trace": copy.deepcopy(self._observed_trace)}

    def restore(self, checkpoint: dict) -> PublicContext:
        if checkpoint.get("schema_version") != "alfworld-prefix-checkpoint.v2":
            raise ValueError("unsupported ALFWorld checkpoint")
        if checkpoint["split"] != self.split or checkpoint["max_steps"] != self.max_steps:
            raise ValueError("ALFWorld checkpoint configuration mismatch")
        self.gamefile = checkpoint["gamefile"]
        self.expected_game_hash = checkpoint["game_hash"]
        self.reset(int(checkpoint["reset_seed"]))
        expected_trace = checkpoint["observed_trace"]
        if not expected_trace or digest(self._observed_trace[0]) != digest(expected_trace[0]):
            raise ValueError("ALFWorld reset observation mismatch")
        for index, action in enumerate(checkpoint["prefix"]):
            if self._terminal:
                raise ValueError("ALFWorld prefix reached terminal state early")
            self.step({"type": "alfworld_text", "text": action})
            if len(expected_trace) <= index + 1 or digest(self._observed_trace[-1]) != digest(expected_trace[index + 1]):
                raise ValueError(f"ALFWorld prefix replay mismatch at step {index}")
        if len(expected_trace) != len(self._observed_trace):
            raise ValueError("ALFWorld replay trace length mismatch")
        return self.observe()

    def state_fingerprint(self) -> str:
        return digest(self.snapshot())

    def terminal_status(self) -> tuple[bool, bool | None]:
        if self._env is None:
            raise RuntimeError("ALFWorld environment is not open")
        return self._terminal, self._success if self._terminal else None

    def validate_action(self, action: dict) -> tuple[bool, str | None]:
        valid = (not self._terminal and isinstance(action, dict) and action.get("type") == "alfworld_text"
                 and isinstance(action.get("text"), str) and bool(action["text"].strip()))
        if not valid:
            return False, "not a valid nonterminal ALFWorld action"
        text = normalize_action(action["text"])
        if self._admissible and text not in self._admissible:
            return False, "action not in current admissible command list"
        return True, None

    def step(self, action: dict, seed: int = 0) -> EnvStep:
        valid, reason = self.validate_action(action)
        if not valid:
            return EnvStep(self.observe(), self._terminal, self._success if self._terminal else None, False, 0.0, reason)
        text = normalize_action(action["text"])
        observations, rewards, dones, infos = self._env.step([text])
        self._observation = str(observations[0])
        self._admissible = tuple(normalize_action(v) for v in _batch(infos, "admissible_commands"))
        won = _scalar(infos, "won", None)
        self._success = bool(won) if won is not None else float(rewards[0]) > 0
        self._terminal = bool(dones[0]) or self._success or len(self._prefix) + 1 >= self.max_steps
        self._prefix.append(text)
        self._observed_trace.append(self._trace_item())
        return EnvStep(self.observe(), self._terminal, self._success if self._terminal else None, True, None)

    def restore_report(self, checkpoint: dict, repeats: int = 3) -> dict:
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        fingerprints, errors = [], []
        for index in range(repeats):
            try:
                self.restore(copy.deepcopy(checkpoint))
                fingerprints.append(self.state_fingerprint())
            except Exception as exc:
                errors.append({"repeat": index, "type": type(exc).__name__, "message": str(exc)})
        return {"repeats": repeats, "observable_replays_matched": not errors and len(set(fingerprints)) == 1,
                "latent_restore_certified": False, "errors": errors, "fingerprints": fingerprints,
                "capabilities": self.capabilities()}

    def close(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
        self._env = None
        self._raw_env = None
