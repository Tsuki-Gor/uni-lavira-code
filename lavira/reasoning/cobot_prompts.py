"""
VLM prompt templates for the Cobot Magic LaViRA iPlanner ObjectNav task.

Every inline prompt string from the original development script is extracted
here as a pure function returning a string. No I/O, no classes. Wording, JSON
contracts, and format markers are preserved verbatim so downstream callers in
``robot/`` and ``tasks/`` produce identical model behaviour.

Exported functions
------------------
get_initial_todo_prompt              – initial TODO checklist generation
get_strategic_decision_prompt        – strategic decide-direction (LA) call
get_tactical_bbox_prompt             – tactical bbox query (VA) call
"""
from __future__ import annotations


def get_initial_todo_prompt(instruction: str) -> str:
    """Header text for initial TODO checklist generation.

    The caller appends the 7-view panorama frames; this function returns
    the leading instruction text and the trailing checklist-format directive is
    available via :func:`get_initial_todo_format`.
    """
    return (
        f'Instruction: "{instruction}"\n\n'
        "Below are panoramic views (7 directions at 45° increments) "
        "from the starting position."
    )


def get_initial_todo_format() -> str:
    """Trailing directive appended after the panorama frames for TODO generation."""
    return (
        "Create a dynamic checklist to complete the instruction.\n"
        "Format as Markdown: - [ ] Step description\n"
        "Return ONLY the checklist. No JSON."
    )


def get_strategic_decision_prompt(
    instruction: str,
    current_todo_list: str = "",
) -> str:
    """Strategic decide-direction (LA) prompt body.

    The caller injects the task/history text and the panorama frames; this
    returns the role/mission/JSON body appended after the frames.
    """
    return f"""
**ROLE**: Intelligent robot navigator.
**MISSION**: "{instruction}"

**Current TODO List**:
{current_todo_list}

**Available directions**: front(0°), left_front(45°), left(90°), left_back(135°),
right_back(-135°), right(-90°), right_front(-45°)

**Tasks**:
1. Update TODO list — mark completed items [x].
2. Decide next action:
   - NAVIGATE: if navigation tasks remain
   - STOP: if all tasks completed

**JSON response** (no markdown, no extra text):
{{
    "progress_analysis": "...(≤30 words)...",
    "reasoning": "...(1-2 sentences)...",
    "updated_todo_list": "...",
    "action": "NAVIGATE" | "STOP",
    "turn_direction": "front|left_front|left|left_back|right_back|right|right_front",
    "expected_landmark": "..."
}}
"""


def get_strategic_history_text(visited_targets: list) -> str:
    """Build the navigation-history block used by the strategic decision prompt."""
    history_info = "Navigation History:\n"
    if visited_targets:
        for i, t in enumerate(visited_targets, 1):
            history_info += (
                f"Step {i}: {t.get('description', 'Unknown')} "
                f"-> {t.get('target', 'Unknown')}\n"
            )
    else:
        history_info += "No history yet.\n"
    return history_info


def get_strategic_task_text(instruction: str, history_info: str) -> str:
    """Leading task + history text placed before the panorama frames (LA call)."""
    return (
        f'Navigation Task: "{instruction}"\n\n'
        f'History:\n{history_info}'
    )


def get_tactical_bbox_prompt(
    instruction: str,
    progress_analysis: str,
    current_target: str,
    rgb_width: int,
    rgb_height: int,
) -> str:
    """Tactical bbox query (VA) prompt body.

    The caller injects the current view image; this returns the role/mission/
    JSON body appended after the image.
    """
    return f"""
**ROLE**: Robot navigator tactical eyes.
**MISSION**: "{instruction}"
**PROGRESS**: "{progress_analysis}"
**CURRENT TARGET**: "{current_target}"

Locate the target object described above.
Coordinates are in pixels [0..{rgb_width}, 0..{rgb_height}].

JSON:
{{
    "visual_check": "I see ...",
    "action": "NAVIGATE" | "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "object name",
    "stop_reasoning": "..."
}}
"""
