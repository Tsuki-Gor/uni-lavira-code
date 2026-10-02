"""
Logging, JSON, and image-encoding helpers used across the codebase.
"""
import os
import json
import re
from typing import Any, Dict, List, Union

import numpy as np

try:
    from colorama import Fore, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:  # pragma: no cover - optional dependency
    class _NoColour:
        RESET = CYAN = GREEN = BLUE = YELLOW = RED = MAGENTA = ""

        def __getattr__(self, _name):
            return ""

    Fore = _NoColour()


# ---------------------------------------------------------------------------
# Coloured stdout logging
# ---------------------------------------------------------------------------

def print_step(step_num: int, description: str):
    """Print step information."""
    print(Fore.CYAN + f"\n[STEP {step_num}] {description}")


def print_action(action: str, details: str = ""):
    """Print action information."""
    print(Fore.GREEN + f"[ACTION] {action}" + (f" - {details}" if details else ""))


def print_info(info: str):
    """Print general information."""
    print(Fore.BLUE + f"[INFO] {info}")


def print_warning(warning: str):
    """Print warning information."""
    print(Fore.YELLOW + f"[WARNING] {warning}")


def print_error(error: str):
    """Print error information."""
    print(Fore.RED + f"[ERROR] {error}")


def print_success(success: str):
    """Print success information."""
    print(Fore.GREEN + f"[SUCCESS] {success}")


def print_model_interaction(
    model_name: str,
    prompt: str,
    response: str,
    speed: float = None,
    duration: float = None,
    prompt_speed: float = None,
):
    """Pretty-print a model call. Truncates base64 image payloads in the prompt."""
    print(Fore.YELLOW + "=" * 60)
    print(Fore.YELLOW + f"MODEL: {model_name}")
    print(Fore.CYAN + "-" * 20 + " PROMPT " + "-" * 20)

    clean_prompt = str(prompt)
    if "data:image" in clean_prompt:
        clean_prompt = re.sub(
            r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+",
            "[IMAGE_BASE64_DATA]",
            clean_prompt,
        )
    print(clean_prompt)

    print(Fore.GREEN + "-" * 20 + " RESPONSE " + "-" * 20)
    print(response)

    if any(v is not None for v in (speed, duration, prompt_speed)):
        print(Fore.MAGENTA + "-" * 20 + " STATS " + "-" * 20)
        if prompt_speed is not None:
            print(Fore.MAGENTA + f"Input speed:  {prompt_speed:.2f} tokens/s")
        if speed is not None:
            print(Fore.MAGENTA + f"Output speed: {speed:.2f} tokens/s")
        if duration is not None:
            print(Fore.MAGENTA + f"Duration:     {duration:.2f}s")
    print(Fore.YELLOW + "=" * 60)


# ---------------------------------------------------------------------------
# Output / image helpers
# ---------------------------------------------------------------------------

def save_output(output_dir: str, filename: str, content: Union[Dict, List, str]):
    """Save output to local file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if isinstance(content, (dict, list)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))
    print_action(f"Saved output file: {filename}")


def numpy_to_base64(img_np: np.ndarray) -> str:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import go1_numpy_to_base64 as shared
    return shared(img_np)


def img_to_base64(img_path: str) -> str:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import go1_img_to_base64 as shared
    return shared(img_path)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import go1_safe_json_loads as shared
    return shared(text)
