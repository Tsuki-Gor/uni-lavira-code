"""
LaViRA Prompt Templates for VLM-based Navigation
==================================================
Prompt templates used by the LA model (strategic planning) and the VA model
(tactical bounding-box detection) for the Unitree G1 humanoid robot.

Note: get_navigation_prompt_image was removed because image_nav is not a
supported task on this platform.
"""


def get_todo_generator_prompt() -> str:
    """Return the system prompt for generating an initial TODO checklist.

    The checklist breaks a high-level navigation instruction into sequential,
    verifiable sub-steps.
    """
    return """
    Your task is to create a dynamic checklist (TODO list) to complete the instruction based on the visual context.

    Requirements:
    - Break down the instruction into logical, sequential steps.
    - Use the visual information to identify landmarks or initial direction if possible.
    - Format as a Markdown checklist:
      - [ ] Step 1 description
      - [ ] Step 2 description

    Response format:
    Return ONLY the markdown checklist string. Do not use JSON.
    """


def get_navigation_prompt_text(
    instruction: str,
    global_target: str,
    current_todo_list: str,
    history_info: str,
    current_step: int = 1,
) -> str:
    """Return the strategic navigation prompt for the LA model.

    The prompt asks the model to update the TODO checklist and choose the
    next movement direction based on the current visual context and history.

    Args:
        instruction: The original natural-language navigation instruction.
        global_target: The overall navigation goal (e.g., "blue chair").
        current_todo_list: Markdown checklist with current progress.
        history_info: Text summary of recent navigation history.
        current_step: Step index (1 = first step, allows "behind" as option).

    Returns:
        Formatted prompt string ready for the LA model.
    """
    allowed_dirs = '"front", "left", "right"'
    if current_step == 1:
        allowed_dirs = '"front", "left", "right", "behind"'

    return f"""
    **ROLE**: You are an intelligent humanoid robot navigator using a generic checklist to guide your actions.
    **MISSION**: "{instruction}"
    **GLOBAL TARGET**: "{global_target}"

    **Current TODO List**:
    {current_todo_list}

    **Task**:
    1. **Update the TODO list**:
       - Check if the current step is completed based on the visual views.
       - If completed, mark it as [x] and append "Result: ...".
       - If the plan is stuck, add new items or modify steps.
    2. **Decide the next action**:
       - Based on the *first incomplete* TODO item.
       - Choose strictly from: {allowed_dirs}.

    **HISTORY ANALYSIS**:
    {history_info}
    (Note: Also refer to the 'Recent Visual History' images provided above for context on movement)

    3. **Stop Decision**:
       - Set "stop": true if you are sure you have reached the final goal.

    **JSON RESPONSE FORMAT**:
    {{
        "progress_analysis": "Assessment of current progress...",
        "updated_todo_list": "The full updated Markdown checklist string (with [x] and [ ])",
        "reasoning": "Why update the list this way and why choose this direction...",
        "turn_direction": "front" or "right" or "left" or "behind",
        "stop": true or false,
        "expected_landmark": "What to look for next"
    }}
    """


def get_tactical_eyes_prompt(
    instruction: str,
    global_target: str,
    strategic_goal: str,
    strategic_stop: bool,
) -> str:
    """Return the tactical perception prompt for the VA model.

    The prompt asks the model to verify the current scene against the
    strategic goal and output a bounding box for the navigation target.

    Args:
        instruction: The original natural-language navigation instruction.
        global_target: The overall navigation goal.
        strategic_goal: The current sub-goal from the strategic planner.
        strategic_stop: Whether the strategic planner has signalled a stop.

    Returns:
        Formatted prompt string ready for the VA model.
    """
    return f"""
    **ROLE**: You are a humanoid robot navigator's TACTICAL EYES.
    **MISSION**: "{instruction}"
    **GLOBAL TARGET**: "{global_target}"
    **CURRENT STRATEGY**: "{strategic_goal}"
    **STRATEGIC STOP SIGNAL**: {strategic_stop}

    **INPUT**: You are looking at the CURRENT VIEW after turning.

    **TASK**:
    1. **Verification**: Do you see the object/area mentioned in "CURRENT STRATEGY"?
    2. **Targeting**: Draw a Bounding Box (bbox_2d) around the best navigation target to move forward.
    - If the GLOBAL TARGET is visible, box it.
    - If not, box the landmark mentioned in CURRENT STRATEGY.
    3. **Action Decision (NAVIGATE vs STOP)**:
    - **NAVIGATE**: If the target is far away or not centered.
    - **STOP**: ONLY if the GLOBAL TARGET is clearly visible, centered, and **occupies more than 20% of the image height**.
    - **SPECIAL CASE**: If STRATEGIC STOP SIGNAL is True, verify if we are indeed at the goal. If yes, output STOP.

    **JSON FORMAT**:
    {{
        "visual_check": "I see [Object] which aligns with strategy...",
        "action": "NAVIGATE" or "STOP",
        "bbox_2d": [x1, y1, x2, y2],
        "target": "Name of the object in the bbox",
        "stop_reasoning": "Only fill this if stopping."
    }}
    """
