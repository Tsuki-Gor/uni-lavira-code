"""
OpenAI-compatible LaViRA API client with two endpoints:

- Language Action (LA) model — strategic planning / direction decisions.
  Maps to ``Config.LA_*``; corresponds to ``client_secondary`` in source.
- Vision Action (VA) model — visual grounding / bounding-box outputs.
  Maps to ``Config.VA_*``; corresponds to ``client`` (primary) in source.

Both default to the same local llama.cpp (llama-server) address so a single model can serve
both roles during development.
"""

import base64
import io
import json
import logging
import os
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


class LaViRA_API:

    def __init__(self, la_api_key=None, la_base_url=None, la_model_name="gpt-4-vision-preview",
                 va_model_name=None, va_api_key=None, va_base_url=None, log=None):
        self.logger = log if log is not None else logger
        self.la_client = OpenAI(
            api_key=la_api_key,
            base_url=la_base_url,
            timeout=2000
        )
        self.la_model_name = la_model_name

        if va_model_name:
            self.va_client = OpenAI(
                api_key=va_api_key,
                base_url=va_base_url,
                timeout=2000

            )
            self.va_model_name = va_model_name
        else:
            self.va_client = None
            self.va_model_name = None

        self.reset_stats()

    def image_to_base64(self, image):
        """Convert a PIL Image or numpy array to a base64-encoded string."""
        from PIL import Image
        import numpy as np

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_debug_info(self, log_path, messages, response_text):
        if not log_path:
            return

        try:
            os.makedirs(log_path, exist_ok=True)

            # Save images and create a clean message list for saving
            saved_messages = []
            img_count = 0

            for msg in messages:
                new_msg = {'role': msg['role'], 'content': []}
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            url = item['image_url']['url']
                            if url.startswith('data:image/'):
                                try:
                                    # Extract base64
                                    header, encoded = url.split(',', 1)
                                    data = base64.b64decode(encoded)
                                    img_filename = f"image_{img_count}.png"
                                    img_path = os.path.join(log_path, img_filename)
                                    with open(img_path, 'wb') as f:
                                        f.write(data)

                                    new_msg['content'].append({
                                        'type': 'image_url',
                                        'image_url': {'url': img_filename}
                                    })
                                    img_count += 1
                                except Exception as e:
                                    self.logger.error(f"Failed to save image: {e}")
                                    new_msg['content'].append({'type': 'image_url', 'image_url': {'url': 'FAILED_TO_SAVE'}})
                            else:
                                new_msg['content'].append(item)
                        else:
                            new_msg['content'].append(item)
                else:
                     new_msg['content'] = msg['content']
                saved_messages.append(new_msg)

            # Save prompt text
            with open(os.path.join(log_path, 'prompt.json'), 'w') as f:
                json.dump(saved_messages, f, indent=2)

            # Save response
            with open(os.path.join(log_path, 'response.txt'), 'w') as f:
                f.write(str(response_text))
        except Exception as e:
            self.logger.error(f"Failed to save debug info: {e}")

    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7, use_la=False, log_path=None, retries=0, max_retries=5, **kwargs):
        """
        Mimic the original model.generate interface.
        Args:
            messages: list of text messages
            images: list of images
            max_new_tokens: max number of tokens to generate
            temperature: sampling temperature
            use_la: whether to use the second (LA) model
            log_path: Path to save debug logs (images and prompt)
            retries: Current retry count
            max_retries: Maximum number of retries
        """
        # use_la = False
        import time
        t = time.time()
        # select the client and model to use
        if use_la and self.la_client:
            client = self.la_client
            model_name = self.la_model_name
            stats_key = 'Language Action Model'
        else:
            client = self.va_client
            model_name = self.va_model_name
            stats_key = 'Vision Action Model'

        # Disable explicit thinking on the Qwen3 family for both LA and VA paths.
        # (Gemini-3.x still reasons internally; this only affects providers that
        # honour the enable_thinking flag, e.g. self-hosted Qwen.)
        extra_body = kwargs.pop('extra_body', None) or {}
        extra_body.setdefault('enable_thinking', False)

        try:
            if stats_key == 'Language Action Model':
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=120,
                    extra_body=extra_body,
                    **kwargs
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=120,
                    reasoning_effort='low',
                    extra_body=extra_body,
                    **kwargs
                )
            self.logger.info(response)
            # Handle case where response is a string (e.g. from some proxies or raw returns)
            if isinstance(response, str):
                self.logger.info(f"API returned string response for {model_name}")
                self.stats[stats_key]['calls'] += 1
                self._save_debug_info(log_path, messages, response)
                return response

            # update usage statistics
            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                self.logger.info(f"API Call usage - {response.usage}")
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0

                self.logger.info(f"{stats_key.upper()} model usage - Input: {response.usage.prompt_tokens}, "
                            f"Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

            self.logger.info(f'Generating uses {time.time() - t} seconds.')
            content = response.choices[0].message.content
            self._save_debug_info(log_path, messages, content)
            return content

        except Exception as e:
            err_str = str(e)
            self.logger.error(f"API Call error with {'secondary' if use_la else 'primary'} model ({model_name}): {e}")
            # Non-recoverable errors — retrying won't help; short-circuit to keep run time bounded.
            non_recoverable = (
                'data_inspection_failed' in err_str or
                'DataInspectionFailed' in err_str or
                'inappropriate content' in err_str or
                'invalid_request_error' in err_str
            )
            if non_recoverable:
                self.logger.error(f"Non-recoverable error detected; skipping retries.")
                return "Error: API rejected request (non-recoverable)"
            if retries >= max_retries:
                self.logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                return "Error: Failed to get response from API after max retries"

            self.logger.info(f'Forcing retry ({retries + 1}/{max_retries})..')
            import time
            time.sleep(30)
            return self.generate(messages, images, max_new_tokens, temperature, use_la, log_path=log_path, retries=retries + 1, max_retries=max_retries, **kwargs)

    def get_model_info(self):
        """Return information about the configured models."""
        info = {
            "primary_model": self.model_name,
            "secondary_model": self.la_model_name if self.secondary_client else None,
            "has_secondary": self.secondary_client is not None
        }
        return info

    def get_usage_stats(self):
        """Return a copy of the current usage statistics."""
        return self.stats.copy()

    def print_usage_stats(self):
        """Log detailed usage statistics and return a summary dict."""
        total_calls = self.stats['Language Action Model']['calls'] + self.stats['Vision Action Model']['calls']
        total_tokens = self.stats['Language Action Model']['total_tokens'] + self.stats['Vision Action Model']['total_tokens']
        if self.la_client:
            self.logger.info("=== MODEL USAGE STATISTICS ===")
            self.logger.info(f"Language Action Model ({self.la_model_name}):")
            self.logger.info(f"  - Calls: {self.stats['Language Action Model']['calls']}")
            self.logger.info(f"  - Input tokens: {self.stats['Language Action Model']['input_tokens']:,}")
            self.logger.info(f"  - Output tokens: {self.stats['Language Action Model']['output_tokens']:,}")
            self.logger.info(f"  - Total tokens: {self.stats['Language Action Model']['total_tokens']:,}")

        if self.va_client:
            self.logger.info(f"Vision Action Model ({self.va_model_name}):")
            self.logger.info(f"  - Calls: {self.stats['Vision Action Model']['calls']}")
            self.logger.info(f"  - Input tokens: {self.stats['Vision Action Model']['input_tokens']:,}")
            self.logger.info(f"  - Output tokens: {self.stats['Vision Action Model']['output_tokens']:,}")
            self.logger.info(f"  - Total tokens: {self.stats['Vision Action Model']['total_tokens']:,}")

        self.logger.info(f"TOTAL:")
        self.logger.info(f"  - Total calls: {total_calls}")
        self.logger.info(f"  - Total tokens: {total_tokens:,}")
        self.logger.info("===============================")

        return {
            'la': self.stats['Language Action Model'].copy(),
            'va': self.stats['Vision Action Model'].copy(),
            'total_calls': total_calls,
            'total_tokens': total_tokens
        }

    def reset_stats(self):
        """Reset usage statistics to zero."""
        self.stats = {
            'Language Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            },
            'Vision Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }

    def eval(self):
        """Compatibility no-op method."""
