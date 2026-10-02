"""
LaViRA navigation API: LA + VA prompt orchestration for the G1 VLN loop.

This module extracts the strategic / tactical VLM orchestration that used to
live inline inside the monolithic source ``tasks/vln.py``.  Behaviour is
preserved verbatim; only the call mechanism changed: the inline OpenAI
completion calls now route through ``LaViRAVisionClient.generate_with_la`` /
``generate_with_va`` with the SAME per-call parameters.

Two roles:

- Language Action (LA) -- strategic / secondary model (source ``client_secondary``
  / Gemini role).  Builds the panorama + visual-history + TODO message and asks
  for the next direction (``language_action``).  ``generate_initial_todo`` is the
  one-shot checklist helper retained from source.
- Vision Action (VA) -- tactical / primary model (source ``client`` / Qwen role).
  Shows the post-rotation view and returns either ``STOP`` or a bounding box of
  the next intermediate target (``vision_action``).

Compatibility behavior inherited from the release
---------------------------------------------------------
The G1 source treated the VA bbox as RAW pixels.  The unified default
``Qwen3.5-27B-Q4`` model emits boxes in the normalised ``[0, 1000]`` range, so
``vision_action`` now de-normalises the bbox into pixel coordinates when the
coordinates look normalised (``max(coords) <= 1000``).  After de-normalisation
the G1 ``cy = bottom-center`` goal-point selection (done in ``tasks/vln.py``) is
preserved unchanged.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from . import g1_prompts as prompts
from ..clients import LaViRAVisionClient
from ._helpers import (
    g1_img_to_base64 as img_to_base64,
    g1_numpy_to_base64 as numpy_to_base64,
    print_action,
    print_error,
    print_info,
    print_model_interaction,
    print_step,
    print_success,
    print_warning,
    g1_safe_json_loads as safe_json_loads,
    save_output,
)


class LaViRANavigationAPI:
    """LA + VA prompt orchestration for the G1 4-direction VLN loop."""

    def __init__(self, client: LaViRAVisionClient, config) -> None:
        self.client = client
        self.config = config

    # ------------------------------------------------------------------ #
    # Language Action
    # ------------------------------------------------------------------ #
    def generate_initial_todo(
        self,
        instruction: str,
        panorama_frames: List[Dict[str, Any]],
    ) -> str:
        """One-shot initial TODO-list generation (LA model).

        Mirrors source ``VLNTask._generate_initial_todo``.  Per-call params:
        LA ``temperature=0.1, max_new_tokens=512``.
        """
        print_action("Generating Initial TODO List...")
        content: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                f'Instruction: "{instruction}"\n\n'
                "The images provided are the 4-directional views from the "
                "starting position."
            ),
        }]
        for frame in panorama_frames:
            if not frame.get("img_base64"):
                print_warning(f"Empty base64 for frame {frame.get('label')}")
                continue
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{frame['img_base64']}"},
            })
            content.append({"type": "text", "text": frame["label"]})

        content.append({"type": "text", "text": prompts.get_todo_generator_prompt()})

        print_action("Sending request to LLM (Strategy - Initial TODO)")
        result_text, stats = self.client.generate_with_la(
            [{"role": "user", "content": content}],
            max_new_tokens=512,
            temperature=0.1,
        )
        todo = result_text.strip()

        print_model_interaction(
            "LA Model (Strategic Brain - Initial TODO)",
            content,
            todo,
            speed=stats.get("output_speed"),
            duration=stats.get("duration"),
            prompt_speed=stats.get("prompt_speed"),
        )

        print_success(f"Initial TODO List:\n{todo}")
        return todo

    def language_action(
        self,
        instruction: str,
        global_target: str,
        todo_list: str,
        iplanner_history: List[str],
        panorama_frames: List[Dict[str, Any]],
        current_step: int,
    ) -> Dict[str, Any]:
        """Strategic-direction call (LA model). Returns the parsed JSON dict.

        Mirrors source ``VLNTask._decide_navigation``.  Per-call params:
        LA ``temperature=0, max_new_tokens=1024``.  Sends the panorama frames,
        the TODO list, and the recent ``replan_`` visual history (sliced to the
        last ``self.config.VISUAL_HISTORY_SIZE`` entries).
        """
        print_step(current_step, "Strategic Brain: Deciding Direction")

        # Prepare History Images: keep only 'replan_' trajectory plans, then
        # take the most recent VISUAL_HISTORY_SIZE of them (source filtered the
        # iplanner history the same way before slicing).
        replan_history = [
            path for path in iplanner_history
            if "replan_" in os.path.basename(path)
        ]
        recent_history_paths = replan_history[-self.config.VISUAL_HISTORY_SIZE:]
        history_content: List[Dict[str, Any]] = []

        if recent_history_paths:
            history_content.append({
                "type": "text",
                "text": "Recent Visual History (Trajectory Plans):",
            })
            for i, path in enumerate(recent_history_paths):
                img_b64 = img_to_base64(path)
                history_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
                history_content.append({
                    "type": "text",
                    "text": f"History Plan -{len(recent_history_paths) - i}",
                })
        else:
            history_content.append({
                "type": "text",
                "text": "No visual history available yet.",
            })

        content: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                f'Navigation Task: "{instruction}"\n\n'
                f"- Current Step: {current_step}"
            ),
        }]

        # Add History Images
        content.extend(history_content)

        content.append({"type": "text", "text": "Current Panorama Views:"})
        for frame in panorama_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{frame['img_base64']}"},
            })
            content.append({"type": "text", "text": frame["label"]})

        prompt = prompts.get_navigation_prompt_text(
            instruction,
            global_target,
            todo_list,
            "Visual History provided above.",
            current_step,
        )
        content.append({"type": "text", "text": prompt})

        print_action("Sending request to LLM (Strategy - Navigation)")
        result_text, stats = self.client.generate_with_la(
            [{"role": "user", "content": content}],
            max_new_tokens=1024,
            temperature=0,
        )

        print_model_interaction(
            "LA Model (Strategic Brain - Navigation)",
            content,
            result_text,
            speed=stats.get("output_speed"),
            duration=stats.get("duration"),
            prompt_speed=stats.get("prompt_speed"),
        )

        save_output(
            self.config.LOG_DIR, f"step{current_step}_navigation_raw.txt", result_text
        )
        return safe_json_loads(result_text)

    # ------------------------------------------------------------------ #
    # Vision Action
    # ------------------------------------------------------------------ #
    def vision_action(
        self,
        img_np,
        instruction: str,
        global_target: str,
        strategic_goal: str,
        strategic_stop: bool,
        current_step: int,
    ) -> Dict[str, Any]:
        """Tactical bbox / NAVIGATE / STOP call (VA model).

        Mirrors source ``VLNTask._query_tactical_eyes`` + ``_draw_bbox``.
        Per-call params: VA ``temperature=0, max_new_tokens=1024``.

        The returned ``bbox_2d`` is de-normalised from the
        ``[0, 1000]`` range emitted by the unified Qwen3.5-27B model into pixel
        coordinates whenever the coordinates look normalised
        (``max(coords) <= 1000``).  The G1 ``cy = bottom-center`` goal point is
        applied later in ``tasks/vln.py`` on the de-normalised pixel box.
        """
        print_step(current_step, "Tactical Eyes: Verifying View")

        if img_np is None:
            return {}

        img_base64 = numpy_to_base64(img_np)
        if not img_base64:
            print_error("Failed to convert image to base64")
            return {}

        prompt = prompts.get_tactical_eyes_prompt(
            instruction, global_target, strategic_goal, strategic_stop
        )

        content: List[Dict[str, Any]] = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            {"type": "text", "text": prompt},
        ]

        print_action("Sending request to LLM (Tactical - Eyes)")
        result_text, stats = self.client.generate_with_va(
            [{"role": "user", "content": content}],
            max_new_tokens=1024,
            temperature=0,
        )

        print_model_interaction(
            "VA Model (Tactical Eyes)",
            content,
            result_text,
            speed=stats.get("output_speed"),
            duration=stats.get("duration"),
            prompt_speed=stats.get("prompt_speed"),
        )

        save_output(self.config.LOG_DIR, f"step{current_step}_bbox_raw.txt", result_text)
        parsed = safe_json_loads(result_text)

        # Preserve the release's bbox conversion from [0, 1000] to pixels.
        # The unified default Qwen3.5-27B-Q4 model emits boxes in the normalised
        # [0, 1000] range; the G1 source assumed raw pixels.  Only scale when the
        # coordinates actually look normalised (max value <= 1000) so already-raw
        # pixel boxes with coordinates above 1000 are left untouched.
        bbox = parsed.get("bbox_2d", [])
        if bbox and len(bbox) == 4:
            try:
                coords = [float(c) for c in bbox]
                if max(coords) <= 1000:
                    img_h, img_w = img_np.shape[:2]
                    x1 = coords[0] / 1000.0 * img_w
                    y1 = coords[1] / 1000.0 * img_h
                    x2 = coords[2] / 1000.0 * img_w
                    y2 = coords[3] / 1000.0 * img_h
                    parsed["bbox_2d"] = [x1, y1, x2, y2]
                    print_info(
                        f"Denormalized BBox [0,1000]->pixel: {parsed['bbox_2d']} "
                        f"(Image size: {img_w}x{img_h})"
                    )
            except (TypeError, ValueError):
                print_warning("BBox coords not numeric; skipping de-normalisation")

        # Save visualization
        save_path = os.path.join(
            self.config.BBOX_DIR, f"step{current_step}_bbox_vis.png"
        )
        self._draw_bbox(img_np.copy(), parsed, save_path)

        return parsed

    # ------------------------------------------------------------------ #
    # Visualisation helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_bbox(img, bbox_data: Dict[str, Any], save_path: str) -> None:
        """Draw the bounding box on the image and save it (source ``_draw_bbox``)."""
        import cv2

        bbox = bbox_data.get("bbox_2d", [])
        if not bbox or len(bbox) != 4:
            cv2.imwrite(save_path, img)
            return

        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        x1, x2 = max(0, int(x1)), min(w, int(x2))
        y1, y2 = max(0, int(y1)), min(h, int(y2))

        action = bbox_data.get("action", "NAVIGATE")
        color = (0, 255, 0) if action == "NAVIGATE" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img,
            f"{action}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.imwrite(save_path, img)
