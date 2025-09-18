#!/usr/bin/env python3
"""
Quick example showing basic robot arm simulation usage.
"""

import sys
import os

# Add the package to path for importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_arm_sim import RobotArm, RobotArmSimulation
import numpy as np


def main():
    print("Robot Arm Simulation - Quick Example")
    print("=" * 40)
    
    # Create a robot arm
    robot = RobotArm(link_lengths=[1.0, 0.8, 0.6])
    print(f"Created robot with {robot.num_joints} joints")
    
    # Test forward kinematics
    robot.set_joint_angles([np.pi/4, np.pi/6, -np.pi/8])
    end_pos, joint_positions = robot.forward_kinematics()
    print(f"End effector position: ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
    
    # Test inverse kinematics
    target_x, target_y = 1.2, 0.8
    print(f"\\nMoving to target: ({target_x}, {target_y})")
    success = robot.move_to_position(target_x, target_y)
    
    if success:
        end_pos, _ = robot.forward_kinematics()
        print(f"✓ Reached: ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
    else:
        print("✗ Target unreachable")
    
    # Create simulation (without visualization for this example)
    sim = RobotArmSimulation(robot)
    print(f"\\nSimulation created with update rate: {sim.update_rate} Hz")
    
    # Simulate a few steps
    print("Running simulation steps...")
    for i in range(5):
        sim.step()
        state = sim.get_state()
        end_pos = state['end_effector_position']
        print(f"  Step {i+1}: End effector at ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
    
    print("\\n✓ Basic robot arm simulation working correctly!")
    print("\\nTo run interactive demos, use: python demo.py")


if __name__ == "__main__":
    main()