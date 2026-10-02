"""Offline request regression against the original release; no model calls."""

import copy
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ("unitree_go1", "unitree_g1", "cobot_magic")
CONFIG = types.SimpleNamespace(
    LA_API_KEY="", LA_BASE_URL="http://la.invalid/v1", LA_MODEL_NAME="la",
    VA_API_KEY="test-key", VA_BASE_URL="http://va.invalid/v1", VA_MODEL_NAME="va",
)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClientRegressionTest(unittest.TestCase):
    def test_shared_client_and_old_entrypoints_preserve_requests(self):
        # A moved import, changed request, retry or failure fallback breaks this.
        self.assertIsNotNone(importlib.util.find_spec("lavira"), "shared package missing")
        sdk = types.ModuleType("openai")
        sdk.OpenAI = None
        sdk_patch = patch.dict(sys.modules, {"openai": sdk})
        sdk_patch.start()
        self.addCleanup(sdk_patch.stop)
        shared = importlib.import_module("lavira.clients")
        baseline = types.ModuleType("original_client")
        source = subprocess.check_output(
            ["git", "show", "215e7ac:real-world-code/unitree_go1/ai_client/vision_client.py"],
            cwd=ROOT, text=True,
        )
        exec(compile(source, "original_client.py", "exec"), baseline.__dict__)
        candidates = [(shared, shared.LaViRAVisionClient)]
        for platform in PLATFORMS:
            module = load_file(platform + "_client", ROOT / "real-world-code" /
                               platform / "ai_client/vision_client.py")
            candidates.append((shared, module.LaViRAVisionClient))

        for use_la, failures, system, usage in (
            (True, 0, False, True), (False, 0, True, True),
            (True, 1, True, True), (False, 2, False, True),
            (True, 3, False, True), (False, 0, False, False),
        ):
            expected = self.capture(baseline, baseline.LaViRAVisionClient,
                                    use_la, failures, system, usage)
            for module, cls in candidates:
                with self.subTest(client=cls.__module__, la=use_la,
                                  failures=failures, system=system, usage=usage):
                    self.assertEqual(self.capture(module, cls, use_la, failures,
                                                  system, usage), expected)

        # Legacy callers may omit config; only the compatibility entrypoint
        # reads platform configuration, while the shared class stays explicit.
        for platform in PLATFORMS:
            module = load_file(platform + "_default", ROOT / "real-world-code" /
                               platform / "ai_client/vision_client.py")
            with patch.dict(sys.modules, {"config": types.SimpleNamespace(Config=CONFIG)}):
                with patch.object(shared, "OpenAI") as factory:
                    client = module.LaViRAVisionClient()
                    self.assertEqual(client.la_model_name, "la")
                    self.assertEqual(factory.call_args_list[0].kwargs,
                                     {"api_key": "no-key", "base_url": CONFIG.LA_BASE_URL})
                    self.assertEqual(factory.call_args_list[1].kwargs,
                                     {"api_key": "test-key", "base_url": CONFIG.VA_BASE_URL})

    def capture(self, module, cls, use_la, failures, system, has_usage):
        requests = []
        constructors = []
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "find the chair"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA="}},
        ]}]
        if system:
            messages.insert(0, {"role": "system", "content": "navigation rules"})
        original_messages = copy.deepcopy(messages)
        usage = types.SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)

        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            if len(requests) <= failures:
                raise RuntimeError("offline simulated API failure")
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"action":"STOP"}'))],
                usage=usage if has_usage else None,
            )

        def sdk(**kwargs):
            constructors.append(kwargs)
            return types.SimpleNamespace(base_url=kwargs["base_url"], chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))

        with patch.object(module, "OpenAI", side_effect=sdk), \
                patch.object(module.time, "time", return_value=10.0), \
                patch.object(module.time, "sleep") as sleep, \
                patch.object(module, "logger"):
            client = cls(CONFIG)
            method = client.generate_with_la if use_la else client.generate_with_va
            try:
                result = method(messages, max_new_tokens=512, temperature=0.1)
            except RuntimeError as exc:
                result = (type(exc).__name__, str(exc))
            stats = copy.deepcopy(client.print_usage_stats())
            info = client.get_model_info()
            client.reset_stats()
            self.assertEqual(client.print_usage_stats()["total_calls"], 0)
            self.assertEqual(messages, original_messages)
            self.assertEqual(len(requests), min(failures + 1, 3))
            self.assertEqual(stats["total_calls"], 0 if failures == 3 else 1)
            return constructors, requests, result, stats, info, sleep.call_args_list


if __name__ == "__main__":
    unittest.main()
