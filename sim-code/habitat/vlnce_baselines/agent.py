"""Build the shared reasoning profile from the existing Habitat configuration."""

import os
from habitat import logger

from lavira.reasoning.habitat import VLMReasoningAgent as SharedReasoningAgent
from .utils.api import LaViRA_API


class VLMReasoningAgent(SharedReasoningAgent):
    def __init__(
        self,
        visualizer,
        task_type: str = "VLN",
        use_guideline=True,
        use_working_memory=True,
        allow_move_behind=True,
        debug_logging=False,
        log_dir="logs/debug_logs",
        use_todo_list=True,
        backtrack_second_chance=True,
        use_backtrack=True,
    ):
        # LA / VA endpoint config is read from environment (see .env.example).
        # Six env vars match the prior LaViRA release naming convention:
        #   LA_API_KEY, LA_BASE_URL, LA_MODEL_NAME, VA_API_KEY, VA_BASE_URL, VA_MODEL_NAME.
        va_api_key    = os.environ.get('VA_API_KEY', '')
        va_base_url   = os.environ.get('VA_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        va_model_name = os.environ.get('VA_MODEL_NAME', 'qwen3.5-27b')

        la_api_key    = os.environ.get('LA_API_KEY', '')
        la_base_url   = os.environ.get('LA_BASE_URL', 'https://yunwu.ai/v1')
        la_model_name = os.environ.get('LA_MODEL_NAME', 'gemini-3.5-flash')
        if not la_api_key or not va_api_key:
            raise RuntimeError(
                "LA_API_KEY and VA_API_KEY must be set as environment variables "
                "(see .env.example). LA = language agent, VA = vision agent."
            )
        logger.info(f"todolist:{use_todo_list}, backtrack_second_chance:{backtrack_second_chance}")
        logger.info(f"LA model={la_model_name} base={la_base_url}")
        logger.info(f"VA model={va_model_name} base={va_base_url}")

        model = LaViRA_API(
            la_api_key=la_api_key,
            la_base_url=la_base_url,
            la_model_name=la_model_name,
            va_model_name=va_model_name,
            va_api_key=va_api_key,
            va_base_url=va_base_url,
        )
        super().__init__(
            model, image_height=visualizer.height, task_type=task_type,
            use_guideline=use_guideline, use_working_memory=use_working_memory,
            allow_move_behind=allow_move_behind, debug_logging=debug_logging,
            log_dir=log_dir, use_todo_list=use_todo_list,
            backtrack_second_chance=backtrack_second_chance, use_backtrack=use_backtrack,
            on_bbox=visualizer._save_rgb_with_bbox, log=logger,
        )
        self.visualizer = visualizer
