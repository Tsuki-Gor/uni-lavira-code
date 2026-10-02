"""
LaViRA navigation API: LA + VA prompt orchestration for the Cobot Magic loop.

This module extracts the strategic / tactical VLM wrappers that used to live
inline inside the original monolithic development script. Behaviour is
preserved verbatim; only the call mechanism changed: the inline primary /
secondary completion calls now route through
``LaViRAVisionClient.generate_with_va`` (tactical / visual-grounding) and
``LaViRAVisionClient.generate_with_la`` (strategic / language) with the SAME
per-call parameters.

Two roles:

- Language Action (LA) — strategic / secondary model. Builds the
  panorama + navigation-history + TODO message and asks for the next direction
  (``decide_direction``). ``generate_initial_todo`` is the one-shot checklist
  helper retained from source.
- Vision Action (VA) — tactical / primary model. Shows the post-rotation view
  and returns either ``STOP`` or a bounding box of the next target. The bbox is
  de-normalised from Qwen's ``[0, 1000]`` range into pixel coordinates
  (``query_llm_bbox``).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from . import cobot_prompts as prompts
from ..clients import LaViRAVisionClient
from ._helpers import (
    image_to_base64,
    print_action,
    print_error,
    print_info,
    print_success,
    go1_safe_json_loads as safe_json_loads,
)


def _debug_enabled() -> bool:
    """Whether per-cycle raw-response ``.txt`` dumps are written.

    Defaults to off. Set ``NAV_DEBUG_DUMP=1`` to enable. Dumps are written into
    the supplied ``cycle_dir`` (never to ``/tmp``); when no ``cycle_dir`` is
    given they are skipped regardless of this flag.
    """
    return os.environ.get("NAV_DEBUG_DUMP", "0") not in ("0", "", "false", "False")


def _dump(cycle_dir: Optional[str], name: str, text: str) -> None:
    """Write a raw model response into ``cycle_dir`` when debug dumps are on."""
    if not cycle_dir or not _debug_enabled():
        return
    try:
        os.makedirs(cycle_dir, exist_ok=True)
        with open(os.path.join(cycle_dir, name), "w") as fh:
            fh.write(text)
    except OSError as exc:  # debug output must never break navigation
        print_error(f"Failed to write debug dump {name}: {exc}")


class LaViRANavigationAPI:
    """LA + VA prompt orchestration for the Cobot Magic ObjectNav / VLN loop."""

    def __init__(self, client: LaViRAVisionClient, config) -> None:
        self.client = client
        self.config = config

    # ------------------------------------------------------------------ #
    # Language Action — strategic
    # ------------------------------------------------------------------ #
    def generate_initial_todo(
        self,
        instruction: str,
        panorama_frames: List[Dict[str, Any]],
    ) -> str:
        """One-shot initial TODO-list generation (LA model).

        Mirrors source ``generate_initial_todo``. Per-call params:
        LA ``max_new_tokens=512, temperature=0.1``.
        """
        print_action("Generating TODO list...")

        content: List[Dict[str, Any]] = []
        content.append({"type": "text", "text": prompts.get_initial_todo_prompt(instruction)})

        for frame in panorama_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame['img_base64']}"},
            })
            content.append({"type": "text", "text": frame["label"]})

        content.append({"type": "text", "text": prompts.get_initial_todo_format()})

        resp, _ = self.client.generate_with_la(
            [{"role": "user", "content": content}],
            max_new_tokens=512, temperature=0.1,
        )
        print_success(f"TODO:\n{resp.strip()}")
        return resp.strip()

    def decide_direction(
        self,
        panorama_frames: List[Dict[str, Any]],
        instruction: str,
        visited_targets: List[Dict[str, Any]],
        current_todo_list: str = "",
        cycle_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Strategic decide-direction call (LA model). Returns the parsed dict.

        Mirrors source ``decide_direction``. Per-call params:
        LA ``max_new_tokens=1024, temperature=0``. Preserves the legacy fallback
        NAVIGATE/right decision when the model errors or returns invalid JSON.
        """
        history_info = prompts.get_strategic_history_text(visited_targets)

        content: List[Dict[str, Any]] = []
        content.append({
            "type": "text",
            "text": prompts.get_strategic_task_text(instruction, history_info),
        })

        for frame in panorama_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame['img_base64']}"},
            })
            content.append({"type": "text", "text": frame["label"]})

        content.append({"type": "text", "text": prompts.get_strategic_decision_prompt(
            instruction, current_todo_list)})

        la_output: Dict[str, Any] = {
            "action": "NAVIGATE",
            "turn_direction": "right",
            "expected_landmark": "open space",
            "updated_todo_list": current_todo_list,
            "progress_analysis": "Fallback",
            "reasoning": "Model error",
        }

        try:
            resp, _ = self.client.generate_with_la(
                [{"role": "user", "content": content}],
                max_new_tokens=1024, temperature=0,
            )
            _dump(cycle_dir, "la_model_raw_response.txt", resp)

            json_str = None
            m = re.search(r"```(?:json)?\s*\n?((?:.|\n)*?)\s*```", resp, re.DOTALL)
            if m:
                json_str = m.group(1)
            else:
                s, e = resp.find("{"), resp.rfind("}")
                if s != -1 and e != -1:
                    json_str = resp[s:e + 1]

            if json_str:
                try:
                    la_output = json.loads(json_str.strip())
                except json.JSONDecodeError as ex:
                    print_info(f"JSON decode error: {ex}\nRaw: {json_str}")

            print_info("LA Output:\n" + json.dumps(la_output, indent=2, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - keep the fallback decision
            print_error(f"Strategic model error: {exc}")

        if cycle_dir and _debug_enabled():
            _dump(cycle_dir, "la_model_output.txt",
                  json.dumps(la_output, indent=2, ensure_ascii=False))

        return la_output

    # ------------------------------------------------------------------ #
    # Vision Action — tactical
    # ------------------------------------------------------------------ #
    def query_llm_bbox(
        self,
        rgb_img,
        instruction: str,
        progress_analysis: str,
        visited_targets: List[Dict[str, Any]],
        strategic_goal: str,
        rgb_width: int,
        rgb_height: int,
        cycle_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Tactical bbox query (VA model) with ``[0, 1000]`` de-normalisation.

        Mirrors source ``query_llm_bbox``. Per-call params:
        VA ``max_new_tokens=1024, temperature=0``. The bbox is de-normalised
        from Qwen's ``[0, 1000]`` range into pixel coordinates and clamped to
        the image bounds. Returns the parsed dict (with pixel ``bbox_2d``) or
        ``None``.
        """
        if rgb_img is None:
            return None

        b64 = image_to_base64(rgb_img)
        current_target = (
            strategic_goal if strategic_goal and len(strategic_goal) > 2
            else instruction
        )

        content: List[Dict[str, Any]] = []
        content.extend([
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompts.get_tactical_bbox_prompt(
                instruction, progress_analysis, current_target,
                rgb_width, rgb_height)},
        ])

        va_output: Optional[Dict[str, Any]] = None
        try:
            resp, _ = self.client.generate_with_va(
                [{"role": "user", "content": content}],
                max_new_tokens=1024, temperature=0,
            )
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if m:
                va_output = json.loads(m.group())

                # Normalise [0,1000] -> pixel coords (Qwen3.5 quirk).
                if (va_output and "bbox_2d" in va_output
                        and isinstance(va_output["bbox_2d"], list)
                        and len(va_output["bbox_2d"]) == 4):
                    b = va_output["bbox_2d"]
                    w, h = rgb_width, rgb_height
                    # Detect if coordinates are in [0,1000] range.
                    if max(b) <= 1000:
                        b = [b[0] / 1000 * w, b[1] / 1000 * h,
                             b[2] / 1000 * w, b[3] / 1000 * h]
                    va_output["bbox_2d"] = [
                        int(max(0, min(w, b[0]))),
                        int(max(0, min(h, b[1]))),
                        int(max(0, min(w, b[2]))),
                        int(max(0, min(h, b[3]))),
                    ]

            print_info("VA Output:\n" + json.dumps(va_output, indent=2, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - waypoint detection is best-effort
            print_error(f"Waypoint error: {exc}")

        if cycle_dir and _debug_enabled():
            _dump(cycle_dir, "va_model_output.txt",
                  json.dumps(va_output or {"error": "No response"},
                             indent=2, ensure_ascii=False))

        return va_output
