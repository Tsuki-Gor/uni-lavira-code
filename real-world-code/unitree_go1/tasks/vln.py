"""Legacy task entrypoint assembling the shared navigation driver."""

from threading import Event

from ai_client import LaViRAVisionClient
from config import Config
from robot.navigation_api import LaViRANavigationAPI
from tasks import register_task
from lavira.agent import LaViRAAgent
from lavira.runtime import Task
from navigation_runtime.camera_robot import CameraRobotRuntime


@register_task("vln")
class VLNTask:
    task_kind = "vln"

    def __init__(self, robot, instruction):
        self.robot = robot
        self.instruction = instruction
        self.global_target = instruction
        self.client = LaViRAVisionClient()
        self.nav_api = LaViRANavigationAPI(self.client)
        self.agent = LaViRAAgent(self.nav_api, profile="go1")
        self.runtime = CameraRobotRuntime(
            robot, Config.PANORAMA_DIR,
            directions=(("front", 0, 0), ("right", 90, -90), ("left", 270, 90)),
            safety_distance=1.5, require_panorama_success=True, recapture_delay=0.5,
            live_cameras=(("front", "camera1"), ("right", "camera2"), ("left", "camera4")))
        self.current_step = 1
        self.todo_list = ""
        self.visited_targets = []
        self.is_task_running = True
        self._cancellation = Event()
        self.result = None

    def stop_task(self):
        self.is_task_running = False
        self._cancellation.set()

    def run(self):
        self.is_task_running = True
        self._cancellation.clear()
        try:
            self.result = self.agent.run(Task(self.instruction, self.task_kind), self.runtime,
                                         cancellation=self._cancellation)
        finally:
            self.current_step = self.agent.state.step
            self.todo_list = self.agent.state.todo_list
            self.visited_targets = self.agent.state.visited_targets
