import unittest

from cave_mem_v2.contracts import PublicContext
from cave_mem_v2.providers import (
    OpenAICompatibleProvider, OpenAIConfig, ProviderActor, ProviderError,
    parse_alfworld_action, render_messages,
)


class ProviderTests(unittest.TestCase):
    def context(self):
        return PublicContext("test", "Perform the task", "visible state", {"phase": "ready"})

    def test_actual_prompt_reminder_is_transient_and_usage_is_measured(self):
        payloads = []
        def transport(payload):
            payloads.append(payload)
            return {"model": "fixed-model", "system_fingerprint": "fp", "choices": [
                {"message": {"content": '{"type":"prepare"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        provider = OpenAICompatibleProvider(OpenAIConfig("fixed-model", send_seed=True,
                                            input_price_per_million=1, output_price_per_million=2), transport=transport)
        actor = ProviderActor(provider)
        remind = actor.propose(self.context(), stage="revision", seed=3, reminder="UNIQUE_REMINDER", call_id="r")
        normal = actor.propose(self.context(), stage="continuation", seed=4, call_id="c")
        self.assertIn("UNIQUE_REMINDER", str(payloads[0]))
        self.assertNotIn("UNIQUE_REMINDER", str(payloads[1]))
        self.assertEqual(remind.usage["input_tokens"], 10)
        self.assertAlmostEqual(remind.usage["api_cost"], 14 / 1e6)
        self.assertEqual(normal.attempts[0]["attempt_id"], "c")
        self.assertEqual(payloads[0]["seed"], 3)
        self.assertFalse(actor.capabilities()["deterministic_sampling_certified"])

    def test_timeout_is_one_metered_attempt_and_no_hidden_retry(self):
        calls = []
        def transport(payload):
            calls.append(payload)
            raise TimeoutError("possibly charged")
        provider = OpenAICompatibleProvider(OpenAIConfig("test"), transport=transport)
        with self.assertRaises(ProviderError) as caught:
            provider.generate(list(render_messages(self.context())), stage="draft", seed=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.attempt["usage"]["physical_attempts"], 1)
        self.assertIsNone(caught.exception.attempt["usage"]["api_cost"])

    def test_missing_usage_and_parameter_aliases(self):
        payloads = []
        def transport(payload):
            payloads.append(payload)
            return {"choices": [{"message": {"content": '{"type":"finish"}'}}]}
        provider = OpenAICompatibleProvider(OpenAIConfig("test", token_limit_parameter="max_completion_tokens"), transport=transport)
        response = provider.generate(list(render_messages(self.context())), stage="draft", seed=1)
        self.assertIsNone(response.usage["input_tokens"])
        self.assertIsNone(response.usage["api_cost"])
        self.assertIn("max_completion_tokens", payloads[0])
        self.assertNotIn("max_tokens", payloads[0])

    def test_reasoning_model_request_omits_temperature(self):
        payloads = []
        def transport(payload):
            payloads.append(payload)
            return {"choices": [{"message": {"content": '{"type":"finish"}'}}]}
        provider = OpenAICompatibleProvider(OpenAIConfig("gpt-5.6-terra", send_temperature=False,
            token_limit_parameter="max_completion_tokens"), transport=transport)
        provider.generate(list(render_messages(self.context())), stage="pilot", seed=1)
        self.assertNotIn("temperature", payloads[0])
        self.assertEqual(payloads[0]["max_completion_tokens"], 512)
        self.assertNotIn("seed", payloads[0])

    def test_alfworld_adapter_preserves_one_command(self):
        self.assertEqual(parse_alfworld_action("Action: put apple 1 in/on fridge 1")["text"], "move apple 1 to fridge 1")
        with self.assertRaises(ValueError):
            parse_alfworld_action("open fridge 1\ntake apple 1")

    def test_malformed_model_action_is_a_metered_behavior_failure(self):
        def transport(payload):
            return {"choices": [{"message": {"content": "not a JSON action"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
        actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig("test"), transport=transport))
        response = actor.propose(self.context(), stage="revision", seed=2)
        self.assertIsNotNone(response.parse_error)
        self.assertEqual(response.action["type"], "__unparsed_action__")
        self.assertEqual(response.usage["input_tokens"], 12)
        self.assertEqual(response.usage["physical_attempts"], 1.)


if __name__ == "__main__":
    unittest.main()
