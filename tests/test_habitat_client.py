"""Preserve the simulator's distinct API protocol without importing Habitat."""

import copy
import importlib
import logging
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class HabitatClientRegressionTest(unittest.TestCase):
    def test_requests_retries_and_error_strings_match_release(self):
        sdk = types.ModuleType("openai")
        sdk.OpenAI = None
        imports = patch.dict(sys.modules, {
            "openai": sdk,
            "PIL": types.SimpleNamespace(Image=None),
            "numpy": types.SimpleNamespace(),
            "habitat": types.SimpleNamespace(logger=logging.getLogger("offline-habitat")),
        })
        imports.start()
        self.addCleanup(imports.stop)
        shared = importlib.import_module("lavira.clients")
        self.assertTrue(hasattr(shared, "LaViRA_API"), "independent simulator client missing")
        with patch.dict(sys.modules, {"habitat": None, "numpy": None, "PIL": None}):
            importlib.reload(shared)
        baseline = types.ModuleType("original_habitat_client")
        source = subprocess.check_output([
            "git", "show", "215e7ac:sim-code/habitat/vlnce_baselines/utils/api.py",
        ], cwd=ROOT, text=True)
        exec(compile(source, "original_habitat_client.py", "exec"), baseline.__dict__)

        legacy_spec = importlib.util.spec_from_file_location(
            "habitat_client_entry", ROOT / "sim-code/habitat/vlnce_baselines/utils/api.py")
        legacy = importlib.util.module_from_spec(legacy_spec)
        legacy_spec.loader.exec_module(legacy)
        with patch.object(shared, "OpenAI"):
            self.assertIs(legacy.LaViRA_API().logger, sys.modules["habitat"].logger)

        for use_la in (True, False):
            for failures, error, raw in (
                (0, "", False), (0, "", True), (1, "temporary", False),
                (6, "temporary", False), (1, "invalid_request_error", False),
            ):
                with self.subTest(la=use_la, failures=failures, error=error, raw=raw):
                    expected = self.capture(baseline, use_la, failures, error, raw)
                    self.assertEqual(self.capture(shared, use_la, failures, error, raw), expected)

    def capture(self, module, use_la, failures, error, raw):
        requests = []
        constructors = []
        messages = [{"role": "user", "content": [{"type": "text", "text": "find the chair"}]}]
        usage = types.SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)

        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            if len(requests) <= failures:
                raise RuntimeError(error)
            if raw:
                return "raw proxy response"
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"action":"STOP"}'))],
                usage=usage,
            )

        def sdk(**kwargs):
            constructors.append(kwargs)
            return types.SimpleNamespace(chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))

        with patch.object(module, "OpenAI", side_effect=sdk), \
                patch("time.time", return_value=10.0), patch("time.sleep") as sleep, \
                patch.object(module, "logger"):
            client = module.LaViRA_API(
                la_api_key="la-key", la_base_url="http://la.invalid/v1", la_model_name="la",
                va_api_key="va-key", va_base_url="http://va.invalid/v1", va_model_name="va",
            )
            result = client.generate(messages, use_la=use_la, max_new_tokens=4096,
                                     temperature=0.0, extra_body={"custom": "initial"})
            stats = copy.deepcopy(client.print_usage_stats())
            client.reset_stats()
            self.assertEqual(client.print_usage_stats()["total_calls"], 0)
            if error == "invalid_request_error":
                self.assertEqual(len(requests), 1)
                self.assertEqual(result, "Error: API rejected request (non-recoverable)")
            elif failures == 6:
                self.assertEqual(len(requests), 6)
                self.assertEqual(result, "Error: Failed to get response from API after max retries")
            else:
                self.assertEqual(stats["total_calls"], 1)
            return constructors, requests, result, stats, sleep.call_args_list


if __name__ == "__main__":
    unittest.main()
