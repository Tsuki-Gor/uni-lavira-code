"""
LaViRA prompt templates for the Unitree Go1 VLN task.

Three prompts are exported:
  get_todo_generator_prompt      – one-shot checklist generation
  get_navigation_prompt_text     – strategic LA navigation call
  get_tactical_eyes_prompt       – tactical VA bbox / NAVIGATE / STOP call
"""


def get_todo_generator_prompt():
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


def get_navigation_prompt_text(instruction, global_target, current_todo_list, history_info, current_step=1):
    return f"""
            **ROLE**: You are an intelligent robot navigator using a generic checklist to guide your actions.
            **MISSION**: "{instruction}"

            **Current TODO List**:
            {current_todo_list}

            **IMPORTANT**: The robot has cameras for 3 directions (Front, Right, Left). Rear camera is unavailable.
            Available directions and their angles:
            - "front" (0°): Straight ahead
            - "left" (90°): Directly left
            - "right" (-90°): Directly right

            **Task**:
            1. **Update the TODO list**:
            - Check if the current step is completed based on the visual views.
            - If the target of the current step is clearly visible and close, mark it as [x] and append "Result: Arrived".
            - If completed, mark it as [x] and append "Result: ...".
            - If the plan is stuck, add new items or modify steps.

            2. **Decide the next action** (CRITICAL - Choose ONE):
            FIRST, analyze the first incomplete item in the updated TODO list to determine the EXACT movement needed:

            - If first incomplete item says "turn left" → use "left" direction
            - If first incomplete item says "turn right" → use "right" direction
            - If first incomplete item says "move forward" → use "front" direction
            - If first incomplete item says "turn" + any direction → use appropriate direction (Avoid 'behind')
            - If multiple navigation tasks remain, focus on the FIRST one

            Then determine the action type:

            - **NAVIGATE**: If there are still navigation tasks (e.g., "find X", "go to Y", "turn Z", "approach W")
                → Return navigation details with direction based on FIRST incomplete TODO item

            - **STOP**: If all navigation tasks are marked [x] completed, or if only non-navigation tasks remain
                → Return "STOP" string

            **CRITICAL**:
            - If you see the target object for the current step is **clearly visible and close** (e.g., occupies a significant part of the view), you MUST mark the current step as [x] and move to the next step or STOP if it's the last step.
            - Do NOT keep navigating to the same target if you are already there.
            - **HOWEVER**: Do not prematurely mark a step as complete if you are still far away or have not even started moving towards it. Only mark complete if you are SURE you have arrived.

            **HISTORY ANALYSIS**:
            {history_info}

            **JSON RESPONSE FORMAT**:
            {{
                "progress_analysis": "One short sentence summarizing current progress (MAX 30 words)",
                "reasoning": "Brief 1-2 sentence explanation why this action and direction was chosen",
                "updated_todo_list": "The full updated Markdown checklist string (with [x] and [ ])",
                "action": "NAVIGATE" or "STOP",
                "turn_direction": "front" or "left" or "right" (ONLY if action is NAVIGATE),
                "expected_landmark": "What to look for next (ONLY if action is NAVIGATE)"
            }}

            **CRITICAL**:
            - progress_analysis and reasoning MUST be extremely concise. Do NOT repeat image descriptions or long analysis.
            - Output ONLY the JSON object.
            - Do NOT output markdown code blocks (```json ... ```).
            - Do NOT output any explanatory text outside the JSON.
            """


def get_tactical_eyes_prompt(instruction, global_target, strategic_goal, strategic_stop, progress_analysis=""):
    # Use strategic goal as the primary target if available, otherwise fallback to instruction
    current_target = strategic_goal if strategic_goal and len(strategic_goal) > 2 else instruction

    return f"""
**ROLE**: You are a robot navigator's TACTICAL EYES.
**MISSION**: "{instruction}"
**PROGRESS ANALYSIS**: "{progress_analysis}"
**CURRENT TARGET**: "{current_target}"

**INPUT**: You are looking at the CURRENT VIEW.

**CRITICAL TASK - OBJECT DETECTION**:
Your PRIMARY goal is to locate and draw a bounding box around: **{current_target}**

**JSON FORMAT**:
{{
    "visual_check": "I see [Object]...",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "Name of the object",
    "stop_reasoning": "Reason if stopping"
}}
"""
