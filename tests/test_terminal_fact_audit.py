"""Public terminal facts need a current, registered diagnostic action as evidence."""
import copy
import unittest

from cave_mem_v2.contracts import PublicContext, canonical
from cave_mem_v2.features import ObservationFactExtractor


DIAGNOSTIC = {"type": "shell", "command": "test -f /app/backup && printf BACKUP_READY"}
RULE = {"source": "history", "diagnostic_actions": [DIAGNOSTIC],
        "diagnostic_basis": "Reviewed public filesystem existence diagnostic for this task scope.",
        "patterns": [{"regex": "BACKUP_READY", "value": True}]}


def event(action, stdout="BACKUP_READY", **overrides):
    result = {"exit_code": 0, "stdout": stdout, "stderr": "", "truncated": False, **overrides}
    return {"action": action, "observation": canonical(result)}


def context(events, *, instruction="Complete the task."):
    return PublicContext("target", instruction, events[-1]["observation"] if events else "ready",
                         history=tuple(events))


class TerminalFactAuditTests(unittest.TestCase):
    def test_registered_diagnostic_establishes_current_public_fact(self):
        extractor = ObservationFactExtractor({"backed_up": RULE})
        self.assertEqual(extractor.extract(context([event(DIAGNOSTIC)])), {"backed_up": True})

    def test_echoed_marker_without_registered_action_is_not_evidence(self):
        extractor = ObservationFactExtractor({"backed_up": RULE})
        self.assertEqual(extractor.extract(context([event({"type": "shell", "command": "echo BACKUP_READY"})])), {})
        forged = PublicContext("target", "Complete the task.", event(DIAGNOSTIC)["observation"])
        self.assertEqual(extractor.extract(forged), {})

    def test_unmatched_action_invalidates_old_diagnostic_even_without_new_marker(self):
        extractor = ObservationFactExtractor({"backed_up": RULE})
        history = [event(DIAGNOSTIC), event({"type": "shell", "command": "rm -f /app/backup"}, "")]
        self.assertEqual(extractor.extract(context(history)), {})

    def test_failed_truncated_or_timed_out_diagnostic_is_unknown(self):
        extractor = ObservationFactExtractor({"backed_up": RULE})
        for update in ({"exit_code": 1}, {"truncated": True}, {"timed_out": True}, {"stdout": None}):
            with self.subTest(update=update):
                self.assertEqual(extractor.extract(context([event(DIAGNOSTIC, **update)])), {})
        malformed = {"action": DIAGNOSTIC, "observation": "BACKUP_READY"}
        self.assertEqual(extractor.extract(context([malformed])), {})

    def test_reviewed_persistence_is_an_exact_action_allowlist(self):
        rule = {**copy.deepcopy(RULE), "persistence": "reviewed_actions",
                "preserve_actions": [{"type": "shell", "command": "pwd"}],
                "persistence_basis": "The exact pwd action cannot modify the backup file in this task scope."}
        extractor = ObservationFactExtractor({"backed_up": rule})
        history = [event(DIAGNOSTIC), event({"type": "shell", "command": "pwd"}, "/app")]
        self.assertEqual(extractor.extract(context(history)), {"backed_up": True})
        history.append(event({"type": "shell", "command": "pwd; rm -f /app/backup"}, "/app"))
        self.assertEqual(extractor.extract(context(history)), {})

    def test_instruction_fact_requires_explicit_static_scope(self):
        rule = {"source": "instruction", "patterns": [{"regex": "requires backup", "value": True}]}
        with self.assertRaises(ValueError):
            ObservationFactExtractor({"requires_preparation": rule})
        extractor = ObservationFactExtractor({"requires_preparation": {**rule, "kind": "task_static"}})
        self.assertEqual(extractor.extract(context([], instruction="This task requires backup.")),
                         {"requires_preparation": True})

    def test_observation_mode_also_requires_action_provenance(self):
        extractor = ObservationFactExtractor({"backed_up": {**RULE, "source": "observation"}})
        self.assertEqual(extractor.extract(context([event(DIAGNOSTIC)])), {"backed_up": True})
        self.assertEqual(extractor.extract(context([event({"type": "shell", "command": "echo BACKUP_READY"})])), {})


if __name__ == "__main__":
    unittest.main()
