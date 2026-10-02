
LA_PROMPT_BACKTRACK = """Based on the navigation history, current 4-directional views, and the TODO list,
decide the next action to find the target object. If appropriate, suggest incremental updates to the TODO list.

Current TODO List:
{todo_list}

{feedback}

Task:
1. FIRST, write `reasoning_todo` and `todo_updates`. Items have only two statuses:
   "pending" and "completed". Review ALL items (not just the first one); a sub-goal
   may be completed out of order.
   - Mark an item `completed` if you judge it done (with a concrete `result`).
   - If marking a LATER item `completed`, re-check EARLIER items.
   - Use `op:rewrite` to refine; `op:add` to introduce a new sub-goal (optional
     `index` to inject at a position; omit to append at end); `op:remove` to drop.
   - `completed` WITHOUT a concrete `result` will be rolled back.

2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO
   plus what you observe.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "progress_analysis": "<brief assessment of what you've observed and done so far>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [
        {{"index": <int>, "status": "completed", "result": "<REQUIRED observation>"}},
        {{"op": "rewrite", "index": <int>, "content": "<refined description>"}},
        {{"op": "add", "content": "<new sub-goal>", "status": "pending", "index": <int (optional; omit to append at end)>}},
        {{"op": "remove", "index": <int>}}
    ],
    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": False|"up"|"down"
}}

Guidelines:
- Do NOT try to open the doors.
- Default to False. Set "up"/"down" ONLY when physically on a staircase crossing between floors. Flat ground, ramps, thresholds, a single step, or carpet edges are all False.
- Set stop=true whenever you have reached the target area needed to answer the question.
  The TODO list does NOT gate stopping — you may stop regardless of how many pending remain.
- Focus on finding the target object specified in the task.
{negative_constraints}"""


LA_PROMPT_BACKTRACK_REPLAN = """You are a navigation agent. You have backtracked to a previous waypoint to have a second chance to choose action.

Instruction: "{instruction}"

Navigation History (up to this waypoint):
(Images provided above)

Current 4-directional views at this waypoint:
(Images provided above)

Previous Action: **navigate to {previous_action}** from here.

Previous Trajectory (Path taken from here):
(Images provided above)

Task:
- Review the Previous Trajectory to understand the outcome of the previous choice.
- Analyze the Current 4-directional views to give a second-chance choice.
- The previous choice can be reconsidered if necessary.

Available Actions:
{action_list}

Response format (JSON):
{{
    "reasoning": "<analysis of the previous path and why the new direction is chosen>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": False|"up"|"down"
}}

Guidelines:
- Do NOT try to open the doors.
- If you believe you have reached the target object at this waypoint, set "stop": True.
- Default to False. Set "up"/"down" ONLY when physically on a staircase crossing between floors. Flat ground, ramps, thresholds, a single step, or carpet edges are all False.
- Focus on finding the target object specified in the task. 
{negative_constraints}"""

LA_PROMPT_TODO_GENERATOR = """You are a navigation agent.
Instruction: "{instruction}"

Create a TODO list to complete this instruction.

Response format (JSON ONLY):
{{
    "todos": [
        {{"content": "<step description>", "status": "pending"}},
        {{"content": "<step description>", "status": "pending"}},
        ...
    ]
}}

Return ONLY the JSON object.
"""

LA_PROMPT_NO_BACKTRACK = """Based on the navigation history, current 4-directional views, and the TODO list,
decide the next navigation direction. If appropriate, suggest incremental updates to the TODO list.

Current TODO List:
{todo_list}

{feedback}

Task:
1. FIRST, write `reasoning_todo` and `todo_updates`. Items have only two statuses:
   "pending" and "completed". Review ALL items (not just the first one); a sub-goal
   may be completed out of order.
   - Mark an item `completed` if you judge it done (with a concrete `result`).
   - If marking a LATER item `completed`, re-check EARLIER items.
   - Use `op:rewrite` to refine; `op:add` to introduce a new sub-goal (optional
     `index` to inject at a position; omit to append); `op:remove` to drop.
   - `completed` without a concrete `result` is rolled back.

2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO
   plus what you observe.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "progress_analysis": "<brief assessment of search progress>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [
        {{"index": <int>, "status": "completed", "result": "<REQUIRED observation>"}},
        {{"op": "rewrite", "index": <int>, "content": "<refined description>"}},
        {{"op": "add", "content": "<new task>", "status": "pending", "index": <int (optional; omit to append)>}},
        {{"op": "remove", "index": <int>}}
    ],
    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",
    "action": "{action_desc}",
    "stop": False,
    "stair": False|"up"|"down"
}}

Guidelines:
- Focus on finding the target object specified in the task.
- Default to False. Set "up"/"down" ONLY when physically on a staircase crossing between floors. Flat ground, ramps, thresholds, a single step, or carpet edges are all False.
{negative_constraints}"""


VA_PROMPT = """Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
- Image size: {width}x{height} pixels
- Current TODO List:
{todo_list}
- Progress Info: {progress_info}

Your task:
1. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
2. Provide the bounding box of the target.

Response format (JSON):
{{
    "reasoning": "<brief explanation of decision>",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- Target description should be specific and clear.
- If deciding to go up or down stairs, the bbox MUST select the ENTRY of the stairs.
- Provide bounding box of next target."""

STOP_CHECK_PROMPT = """You are an intelligent navigation agent.
Task: "{instruction}"

You have decided to STOP, believing you have reached the target.
Now, please double-check your decision based on the current 4-directional views and the previous waypoint's 4-directional views.

Previous Waypoint Views:
{previous_views}

Current Views:
{current_views}

Requirements:
1. Check if the target object is clearly visible in any of the views.
2. Estimate if the distance to the target is less than 1 meter.
3. Compare previous views with current views to confirm you have approached the target.

Decision Rules:
- If target is visible AND distance < 1m: CONFIRM STOP.
- If target is NOT visible OR distance >= 1m: CONTINUE NAVIGATION.

Response format (JSON):
{{
    "analysis": "<analysis of visibility and distance, and comparison with previous views>",
    "decision": "STOP" or "CONTINUE"
}}
"""


# EQA Oracle: prepended as the leading text block when querying the LA model at the
# destination panorama. The `{instruction}` placeholder is the EQA question.
ORACLE_SYSTEM_PROMPT = (
    'You are a robot agent. You have reached your destination.\n'
    'You have access to:\n'
    'A set of 4 Panorama Views at your current location (Destination).\n\n'
    'Task: Answer the question: "{instruction}"'
)


# EQA Oracle QA: appended after the panorama image blocks. Restricts the answer to
# the allowed vocabulary and forces a guess so the LA never returns "null".
EQA_QA_PROMPT = """
Task: You are a visual scene analyzer. Your ONLY job is to select the single best answer from the provided lists based ONLY on the CURRENT location views.
Question: "{instruction}"

**Allowed Vocabulary:**
- **COLORS:** ["white", "off-white", "black", "grey", "blue", "red", "green", "brown", "yellow", "slate grey", "tan", "silver", "red brown", "olive green"]
- **ROOMS:** ["kitchen", "bedroom", "spa", "lounge", "family room", "living room", "hallway", "laundry room", "gym", "closet"]

**EXTREME RULES (MUST FOLLOW):**
1. **FORCE A GUESS:** Even if the target is invisible, occluded, or ambiguous, you **MUST HALLUCINATE or GUESS** based on the scene context.
2. **STRICT SELECTION:** Your output "answer" MUST be **EXACTLY** one of the words from the "Allowed Vocabulary" lists above.

Response format (JSON):
{{
    "reasoning": "Briefly explain...",
    "answer": "<THE SELECTED WORD>"
}}
"""



# Leading text block passed to the VA model alongside the current RGB and depth
# images. Kept as a constant so the system prompt lives next to the rest of the
# task prompts. No placeholders.
ROBOT_NAVIGATION_SYSTEM_PROMPT = (
    "You are a robot performing navigation task. "
    "Look at this image and the corresponding depth gray image and help the robot navigate."
)
