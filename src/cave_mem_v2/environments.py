"""Environment contract and a fully restorable mechanism/action factorial fixture.

Hidden generation controls exist only in snapshots. Public facts are explicit
observable task state. The fixture validates collectors; scripted rescues or
regressions are not empirical claims about real language models.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import PublicContext, canonical, digest, finite


@dataclass(frozen=True)
class EnvStep:
    context: PublicContext
    terminal: bool
    success: bool | None
    valid: bool = True
    tool_cost: float | None = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if type(self.terminal) is not bool or type(self.valid) is not bool:
            raise ValueError("terminal and valid must be boolean")
        if self.success is not None and type(self.success) is not bool:
            raise ValueError("success must be boolean or unknown")
        if self.terminal and type(self.success) is not bool:
            raise ValueError("terminal environment result needs explicit success")
        if self.tool_cost is not None:
            finite(self.tool_cost, "tool_cost", minimum=0)


class Environment(Protocol):
    def reset(self, seed: int = 0) -> PublicContext: ...
    def observe(self) -> PublicContext: ...
    def snapshot(self) -> dict: ...
    def restore(self, checkpoint: dict) -> PublicContext: ...
    def state_fingerprint(self) -> str: ...
    def configuration(self) -> dict: ...
    def capabilities(self) -> dict: ...
    def terminal_status(self) -> tuple[bool, bool | None]: ...
    def validate_action(self, action: dict) -> tuple[bool, str | None]: ...
    def step(self, action: dict, seed: int = 0) -> EnvStep: ...
    def close(self) -> None: ...


class FactorialEnvironment:
    """A task with independent mechanism applicability and initial action intent.

    `applicable` changes whether operating before preparation destroys success.
    `safe_intent` controls an ordinary, visible instruction at the checkpoint;
    it does not pass a privileged actor-profile flag to model features.
    Each mechanism shares a small action vocabulary so a generic collector can
    be exercised on held-out visible templates/mechanisms. Factory callers own
    and declare the corresponding split groups.
    """

    MECHANISMS = {
        "backup": ("Create and validate a backup", "modify the resource", "verify the restored resource"),
        "dependency": ("Install and verify the dependency", "run the build", "verify the built artifact"),
        "lock": ("Acquire the required lock", "commit the update", "verify the committed record"),
        "checksum": ("Validate the input checksum", "process the input", "verify the output"),
    }

    def __init__(self, *, task_id: str = "factorial-task", applicable: bool = True,
                 safe_intent: bool = False, hide_applicability: bool = False,
                 mechanism: str = "backup", template: str = "direct", max_steps: int = 8,
                 tool_cost: float = 0.0) -> None:
        if any(type(v) is not bool for v in (applicable, safe_intent, hide_applicability)):
            raise ValueError("factorial controls must be boolean")
        if mechanism not in self.MECHANISMS or template not in ("direct", "checklist", "compact"):
            raise ValueError("unknown fixture mechanism/template")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive integer")
        finite(tool_cost, "tool_cost", minimum=0)
        self.task_id = task_id
        self.applicable = applicable
        self.safe_intent = safe_intent
        self.hide_applicability = hide_applicability
        self.mechanism = mechanism
        self.template = template
        self.max_steps = max_steps
        self.tool_cost = tool_cost
        self._state: dict[str, Any] | None = None
        self._history: list[dict] = []

    def configuration(self) -> dict:
        # Per-task hidden generation knobs are intentionally absent from config
        # presented to the actor/feature pipeline. Snapshot contains them.
        return {"adapter": "factorial.v2", "dynamics": "deterministic",
                "scoring": "binary_terminal_success", "max_steps": self.max_steps,
                "tool_cost_per_execution": self.tool_cost}

    def capabilities(self) -> dict:
        return {"restore": "exact_state", "restore_certified": True, "state_fingerprint_scope": "complete_fixture_state",
                "future_randomness": "deterministic", "step_seed_applied": True,
                "isolated_sessions": True, "read_only_diagnostics": ("requires_preparation",),
                "research_role": "fixture_only"}

    def reset(self, seed: int = 0) -> PublicContext:
        if type(seed) is not int:
            raise ValueError("seed must be integer")
        self._state = {"phase": "ready", "prepared": False, "terminal": False,
                       "success": False, "steps": 0}
        self._history = []
        return self.observe()

    def _require_state(self) -> dict:
        if self._state is None:
            raise RuntimeError("reset or restore the environment before use")
        return self._state

    def observe(self) -> PublicContext:
        state = self._require_state()
        prep, operate, finish = self.MECHANISMS[self.mechanism]
        instruction = f"Complete the task: {operate}, then {finish}."
        if self.safe_intent:
            instruction += f" Your ordinary current plan starts with: {prep}."
        if self.template == "checklist":
            instruction = "Checklist task. " + instruction
        elif self.template == "compact":
            instruction = "Goal: " + instruction
        facts = {"phase": state["phase"], "prepared": state["prepared"],
                 "mechanism": self.mechanism, "intent": "prepare" if self.safe_intent else "operate"}
        if not self.hide_applicability:
            facts["requires_preparation"] = self.applicable
        admissible = ("prepare", "operate") if state["phase"] == "ready" else ("finish",)
        if state["terminal"]:
            admissible = ()
        observation = f"phase={state['phase']}; preparation_completed={state['prepared']}"
        if not self.hide_applicability:
            observation += f"; preparation_required={self.applicable}"
        return PublicContext(self.task_id, instruction, observation, facts,
                             tuple(copy.deepcopy(self._history)), admissible,
                             max(0, self.max_steps - state["steps"]))

    def snapshot(self) -> dict:
        return {"schema_version": "factorial-checkpoint.v2", "task_id": self.task_id,
                "controls": {"applicable": self.applicable, "safe_intent": self.safe_intent,
                             "hide_applicability": self.hide_applicability, "mechanism": self.mechanism,
                             "template": self.template, "max_steps": self.max_steps, "tool_cost": self.tool_cost},
                "state": copy.deepcopy(self._require_state()), "history": copy.deepcopy(self._history)}

    def restore(self, checkpoint: dict) -> PublicContext:
        if checkpoint.get("schema_version") != "factorial-checkpoint.v2":
            raise ValueError("unsupported factorial checkpoint")
        controls = checkpoint["controls"]
        validated = FactorialEnvironment(task_id=checkpoint["task_id"], **controls)
        self.__dict__.update(validated.__dict__)
        state = copy.deepcopy(checkpoint["state"])
        if set(state) != {"phase", "prepared", "terminal", "success", "steps"}:
            raise ValueError("unexpected checkpoint state")
        if state["phase"] not in ("ready", "operated", "finished", "failed"):
            raise ValueError("invalid checkpoint phase")
        if any(type(state[k]) is not bool for k in ("prepared", "terminal", "success")):
            raise ValueError("invalid checkpoint truth values")
        if type(state["steps"]) is not int or not 0 <= state["steps"] <= self.max_steps:
            raise ValueError("invalid checkpoint step count")
        self._state = state
        self._history = copy.deepcopy(checkpoint["history"])
        canonical(self.snapshot())
        return self.observe()

    def state_fingerprint(self) -> str:
        return digest(self.snapshot())

    def terminal_status(self) -> tuple[bool, bool | None]:
        state = self._require_state()
        return state["terminal"], state["success"] if state["terminal"] else None

    def validate_action(self, action: dict) -> tuple[bool, str | None]:
        state = self._require_state()
        if state["terminal"]:
            return False, "environment is terminal"
        allowed = ("prepare", "operate") if state["phase"] == "ready" else ("finish",)
        valid = isinstance(action, dict) and set(action) == {"type"} and action.get("type") in allowed
        return valid, None if valid else "operation not allowed in current phase"

    def step(self, action: dict, seed: int = 0) -> EnvStep:
        valid, reason = self.validate_action(action)
        if not valid:
            return EnvStep(self.observe(), False, None, False, 0.0, reason)
        state = self._require_state()
        operation = action["type"]
        state["steps"] += 1
        if operation == "prepare":
            state["prepared"] = True
        elif operation == "operate":
            if self.applicable and not state["prepared"]:
                state.update(phase="failed", terminal=True, success=False)
            else:
                state["phase"] = "operated"
        elif operation == "finish":
            state.update(phase="finished", terminal=True, success=True)
        if not state["terminal"] and state["steps"] >= self.max_steps:
            state.update(terminal=True, success=False)
        public_result = f"executed={operation}; phase={state['phase']}"
        self._history.append({"action": copy.deepcopy(action), "observation": public_result})
        return EnvStep(self.observe(), state["terminal"], state["success"] if state["terminal"] else None,
                       True, self.tool_cost)

    def inspect(self, name: str) -> dict:
        if name != "requires_preparation":
            raise ValueError("unapproved diagnostic")
        before = self.state_fingerprint()
        result = {"requires_preparation": self.applicable}
        if before != self.state_fingerprint():
            raise RuntimeError("diagnostic changed environment state")
        return result

    def close(self) -> None:
        pass
