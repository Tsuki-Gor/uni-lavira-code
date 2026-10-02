"""Robot layer for the Unitree G1 humanoid (controller, iPlanner client, navigation)."""
from .iplanner_client import IPlannerRemoteClient
from .robot_controller import RobotController

__all__ = [
    "RobotController",
    "IPlannerRemoteClient",
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
