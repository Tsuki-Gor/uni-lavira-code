"""Robot layer for Cobot Magic (hardware + motion + navigation).

Exposes the dual-arm mobile-base controller, the panorama arm gimbal, the
in-process iPlanner client, the LA/VA VLM navigation API, and the thin
integrated vision-navigation controller used by the web / demo entry point.
"""
from robot.arm_controller import AutomatedController
from robot.iplanner_client import IPlannerClient
from robot.robot_controller import RobotController

__all__ = [
    "RobotController",
    "AutomatedController",
    "IPlannerClient",
    "LaViRANavigationAPI",
    "IntegratedVisionNavController",
]


def __getattr__(name):
    # Importing hardware alone must not load the navigation algorithm.
    from importlib import import_module

    modules = {"LaViRANavigationAPI": ".navigation_api",
               "IntegratedVisionNavController": ".nav_controller"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
