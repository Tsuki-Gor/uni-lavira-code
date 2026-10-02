"""
OpenAI-compatible LaViRA API client with two endpoints:

- Language Action (LA) model — strategic planning / direction decisions.
  Maps to ``Config.LA_*``; corresponds to ``client_secondary`` in source.
- Vision Action (VA) model — visual grounding / bounding-box outputs.
  Maps to ``Config.VA_*``; corresponds to ``client`` (primary) in source.

Both default to the same local llama.cpp (llama-server) address so a single model can serve
both roles during development.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 10.0


class LaViRAVisionClient:
    """Two-endpoint OpenAI-compatible client for the LaViRA real-world stack.

    Parameters
    ----------
    config:
        Object exposing ``LA_API_KEY``, ``LA_BASE_URL``, ``LA_MODEL_NAME``,
        ``VA_API_KEY``, ``VA_BASE_URL``, ``VA_MODEL_NAME``. Configuration is
        supplied by the caller; this module does not import platform code.
    """

    def __init__(self, config) -> None:
        # Language-Action (LA) — strategic / secondary
        self.la_client = OpenAI(
            api_key=config.LA_API_KEY or "no-key",
            base_url=config.LA_BASE_URL,
        )
        self.la_model_name: str = config.LA_MODEL_NAME

        # Vision-Action (VA) — tactical / primary
        self.va_client = OpenAI(
            api_key=config.VA_API_KEY or "no-key",
            base_url=config.VA_BASE_URL,
        )
        self.va_model_name: str = config.VA_MODEL_NAME

        self.stats: Dict[str, Dict[str, int]] = {
            "Language Action Model": {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
            },
            "Vision Action Model": {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
            },
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepend_no_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepend ``/no_think`` to suppress Qwen3 chain-of-thought tokens."""
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            first = dict(messages[0])
            first["content"] = "/no_think " + first["content"]
            messages[0] = first
        else:
            messages.insert(0, {"role": "system", "content": "/no_think"})
        return messages

    def _call_once(
        self,
        client: OpenAI,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        stats_key: str,
    ) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        duration = time.time() - t0
        usage = getattr(response, "usage", None)

        self.stats[stats_key]["calls"] += 1
        if usage:
            self.stats[stats_key]["input_tokens"] += usage.prompt_tokens or 0
            self.stats[stats_key]["output_tokens"] += usage.completion_tokens or 0
            self.stats[stats_key]["total_tokens"] += usage.total_tokens or 0

        prompt_speed = (
            (usage.prompt_tokens / duration)
            if usage and usage.prompt_tokens and duration > 0 else 0.0
        )
        output_speed = (
            (usage.completion_tokens / duration)
            if usage and usage.completion_tokens and duration > 0 else 0.0
        )

        content = response.choices[0].message.content
        return content, {
            "duration": duration,
            "prompt_speed": prompt_speed,
            "output_speed": output_speed,
            "usage": usage,
        }

    # ------------------------------------------------------------------ #
    # Core generate
    # ------------------------------------------------------------------ #

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        use_la: bool = False,
        _attempt: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a multimodal chat completion and return ``(content, info_dict)``.

        Parameters
        ----------
        messages:
            OpenAI-format message list.  Embed images as ``image_url`` blocks
            with ``data:image/jpeg;base64,...`` URLs (use ``utils.numpy_to_base64``
            or ``utils.img_to_base64`` for encoding).
        max_new_tokens:
            Max tokens to generate.  Source values: 512 for initial TODO,
            1024 for navigation / tactical / EQA calls.
        temperature:
            Source values: 0.1 for initial TODO, 0.0 for all other calls.
        use_la:
            *True* routes to the Language-Action model; *False* to Vision-Action.
        """
        client, model_name, stats_key = (
            (self.la_client, self.la_model_name, "Language Action Model")
            if use_la else
            (self.va_client, self.va_model_name, "Vision Action Model")
        )

        messages = self._prepend_no_think(messages)

        try:
            return self._call_once(
                client, model_name, messages, max_new_tokens, temperature, stats_key
            )
        except Exception as exc:
            logger.error(
                "[LaViRAVisionClient] API error (%s / %s): %s",
                stats_key, model_name, exc,
            )
            logger.debug("Traceback for the above error:", exc_info=True)

            if _attempt >= _MAX_RETRIES - 1:
                raise

            time.sleep(_RETRY_DELAY)
            return self.generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_la=use_la,
                _attempt=_attempt + 1,
            )

    def generate_with_la(
        self, messages: List[Dict[str, Any]], **kw: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """Convenience wrapper: always uses the Language-Action (LA) model."""
        return self.generate(messages, use_la=True, **kw)

    def generate_with_va(
        self, messages: List[Dict[str, Any]], **kw: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """Convenience wrapper: always uses the Vision-Action (VA) model."""
        return self.generate(messages, use_la=False, **kw)

    # ------------------------------------------------------------------ #
    # Usage statistics
    # ------------------------------------------------------------------ #

    def get_model_info(self) -> Dict[str, Any]:
        """Return a dict with the configured model names and base URLs."""
        return {
            "la_model": self.la_model_name,
            "la_base_url": self.la_client.base_url,
            "va_model": self.va_model_name,
            "va_base_url": self.va_client.base_url,
        }

    def print_usage_stats(self) -> Dict[str, Any]:
        """Print and return cumulative token usage for both models."""
        la = self.stats["Language Action Model"]
        va = self.stats["Vision Action Model"]
        total_calls = la["calls"] + va["calls"]
        total_tokens = la["total_tokens"] + va["total_tokens"]

        logger.info("=== MODEL USAGE STATISTICS ===")
        logger.info("Language Action (%s):", self.la_model_name)
        logger.info("  Calls: %d  |  Tokens: %s", la["calls"], f"{la['total_tokens']:,}")
        logger.info("Vision Action (%s):", self.va_model_name)
        logger.info("  Calls: %d  |  Tokens: %s", va["calls"], f"{va['total_tokens']:,}")
        logger.info("TOTAL: %d calls, %s tokens", total_calls, f"{total_tokens:,}")
        logger.info("==============================")

        return {
            "la": la.copy(), "va": va.copy(),
            "total_calls": total_calls, "total_tokens": total_tokens,
        }

    def reset_stats(self) -> None:
        """Reset all per-model usage counters to zero."""
        for key in self.stats:
            self.stats[key] = {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
            }
