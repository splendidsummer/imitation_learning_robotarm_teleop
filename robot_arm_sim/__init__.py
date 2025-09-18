"""
Robot Arm Simulation Package

A simple robot arm simulation environment for imitation learning and teleoperation.
"""

__version__ = "0.1.0"
__author__ = "Robot Arm Simulation Team"

from .robot_arm import RobotArm
from .simulation import RobotArmSimulation

__all__ = ["RobotArm", "RobotArmSimulation"]