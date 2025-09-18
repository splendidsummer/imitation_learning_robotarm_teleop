"""Robot Arm Teleoperation Package

This package provides simulation environments and teleoperation control
for robot arms, implementing ALOHA and other configurations.
"""

__version__ = "0.1.0"
__author__ = "splendidsummer"

__all__ = [
    "RobotArmSimulation",
    "TeleoperationController", 
    "ALOHAEnvironment",
]

def __getattr__(name):
    if name == "RobotArmSimulation":
        from .simulation import RobotArmSimulation
        return RobotArmSimulation
    elif name == "TeleoperationController":
        from .teleoperation import TeleoperationController
        return TeleoperationController
    elif name == "ALOHAEnvironment":
        from .environments import ALOHAEnvironment
        return ALOHAEnvironment
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")