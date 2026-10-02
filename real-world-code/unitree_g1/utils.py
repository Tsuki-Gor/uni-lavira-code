"""
LaViRA Utility Functions for Unitree G1
========================================
Shared utilities for logging, image processing, and JSON parsing.
"""

import os
import json
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:  # pragma: no cover - optional dependency
    class _NoColour:
        RESET = CYAN = GREEN = BLUE = YELLOW = RED = MAGENTA = WHITE = ""

        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = _NoColour()
    Style = _NoColour()


# =========================================================================
# Logging Utilities
# =========================================================================

def print_step(step_num: int, description: str) -> None:
    """Print step information."""
    print(Fore.CYAN + f"\n{'='*60}")
    print(Fore.CYAN + f"[STEP {step_num}] {description}")
    print(Fore.CYAN + f"{'='*60}")


def print_action(action: str, details: str = "") -> None:
    """Print action information."""
    print(Fore.GREEN + f"  [ACTION] {action}" + (f" - {details}" if details else ""))


def print_info(info: str) -> None:
    """Print general information."""
    print(Fore.BLUE + f"  [INFO] {info}")


def print_warning(warning: str) -> None:
    """Print warning information."""
    print(Fore.YELLOW + f"  [WARNING] {warning}")


def print_error(error: str) -> None:
    """Print error information."""
    print(Fore.RED + f"  [ERROR] {error}")


def print_success(success: str) -> None:
    """Print success information."""
    print(Fore.GREEN + f"  [SUCCESS] {success}")


def print_model_response(response_type: str, content: str) -> None:
    """Print model response information."""
    print(Fore.MAGENTA + f"  [MODEL {response_type}] {content}")


def print_robot(message: str) -> None:
    """Print G1 robot-specific status."""
    print(Fore.WHITE + f"  [G1 ROBOT] {message}")


def print_model_interaction(
    model_name: str,
    prompt: str,
    response: str,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    prompt_speed: Optional[float] = None,
) -> None:
    """Pretty-print a model call.

    Truncates base64 image payloads in the prompt for readability.

    Args:
        model_name: Name of the model being called.
        prompt: The full prompt sent to the model.
        response: The model's response text.
        speed: Output token generation speed (tokens/s), if available.
        duration: Total call duration in seconds, if available.
        prompt_speed: Input token processing speed (tokens/s), if available.
    """
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


# =========================================================================
# File I/O Utilities
# =========================================================================

def save_output(output_dir: str, filename: str, content: Union[Dict, List, str]) -> None:
    """Save output to a local file.

    Args:
        output_dir: Directory path where the file will be written.
        filename: Name of the output file.
        content: Data to write; dicts and lists are serialised as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if isinstance(content, (dict, list)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))
    print_action(f"Saved output file: {filename}")


# =========================================================================
# Image Utilities
# =========================================================================

def numpy_to_base64(img_np: np.ndarray) -> str:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import g1_numpy_to_base64 as shared
    return shared(img_np)


def img_to_base64(img_path: str) -> str:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import g1_img_to_base64 as shared
    return shared(img_path)


# =========================================================================
# JSON Parsing Utilities
# =========================================================================

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import g1_safe_json_loads as shared
    return shared(text)
