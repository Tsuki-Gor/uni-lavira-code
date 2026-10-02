"""Cobot task assembly for the shared navigation driver."""

import datetime
import os
from threading import Event

from ai_client import LaViRAVisionClient
from config import Config
from robot.navigation_api import LaViRANavigationAPI
from tasks import register_task
from lavira.agent import LaViRAAgent
from lavira.runtime import Task
from navigation_runtime.gimbal_robot import GimbalRobotRuntime


@register_task("vln")
class VLNTask:
    task_kind = "vln"

    def __init__(self, robot, instruction, max_cycles=10):
        self.robot = robot
        self.instruction = instruction or "navigate forward"
        self.max_cycles = max_cycles
        self.client = LaViRAVisionClient()
        self.nav_api = LaViRANavigationAPI(self.client)
        self.agent = LaViRAAgent(self.nav_api, profile="cobot")
        self.visited_targets = []
        self.current_todo_list = ""
        self.is_task_running = True
        self._cancellation = Event()
        self.result = None

    def stop_task(self):
        self.is_task_running = False
        self._cancellation.set()

    def run(self):
        self.is_task_running = True
        self._cancellation.clear()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        session_root = Config.SESSION_DIR or os.path.join(Config.OUTPUT_ROOT, timestamp)
        os.makedirs(session_root, exist_ok=True)
        runtime = GimbalRobotRuntime(self.robot, session_root,
                                     include_depth=self.task_kind == "object_nav")
        try:
            self.result = self.agent.run(
                Task(self.instruction, self.task_kind, max_cycles=max(0, self.max_cycles)),
                runtime, cancellation=self._cancellation)
        finally:
            self.visited_targets = self.agent.state.visited_targets
            self.current_todo_list = self.agent.state.todo_list
