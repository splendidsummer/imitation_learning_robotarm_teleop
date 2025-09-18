#!/usr/bin/env python3
"""
Basic robot arm simulation example.

This script demonstrates how to use the RobotArmSimulation class
to create and control a robot arm in a PyBullet environment.
"""

import sys
import os
import time

# Add the package to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm_teleop.simulation import RobotArmSimulation
import numpy as np


def main():
    """Run basic robot arm simulation example."""
    print("Starting basic robot arm simulation example...")
    
    # Create robot arm simulation
    robot_sim = RobotArmSimulation(use_gui=True)
    
    try:
        # Print robot information
        print(f"Robot has {len(robot_sim.joint_indices)} controllable joints:")
        for i, name in enumerate(robot_sim.joint_names):
            print(f"  Joint {i}: {name}")
        
        # Demonstrate basic movements
        print("\\nDemonstrating basic movements...")
        
        # Get current joint positions
        current_positions, _ = robot_sim.get_joint_states()
        print(f"Current joint positions: {[f'{pos:.3f}' for pos in current_positions]}")
        
        # Move to a target position
        if len(robot_sim.joint_indices) >= 3:
            target_positions = current_positions.copy()
            target_positions[0] = 0.5  # Move first joint
            target_positions[1] = -0.3  # Move second joint
            if len(target_positions) > 2:
                target_positions[2] = 0.8  # Move third joint
            
            print(f"Moving to target: {[f'{pos:.3f}' for pos in target_positions]}")
            robot_sim.move_to_joint_positions(target_positions)
            
            # Let the robot move
            for _ in range(100):
                robot_sim.step_simulation()
                time.sleep(0.01)
        
        # Get end effector pose
        ee_pos, ee_ori = robot_sim.get_end_effector_pose()
        print(f"End effector position: {[f'{pos:.3f}' for pos in ee_pos]}")
        print(f"End effector orientation: {[f'{ori:.3f}' for ori in ee_ori]}")
        
        # Demonstrate inverse kinematics
        if robot_sim.end_effector_link is not None:
            print("\\nDemonstrating inverse kinematics...")
            target_ee_pos = [0.3, 0.2, 0.8]  # Target position
            
            try:
                ik_solution = robot_sim.inverse_kinematics(target_ee_pos)
                print(f"IK solution: {[f'{pos:.3f}' for pos in ik_solution]}")
                
                # Move to IK solution
                robot_sim.move_to_joint_positions(ik_solution)
                
                # Let the robot move
                for _ in range(100):
                    robot_sim.step_simulation()
                    time.sleep(0.01)
                
                # Check if we reached the target
                final_ee_pos, _ = robot_sim.get_end_effector_pose()
                error = np.linalg.norm(np.array(final_ee_pos) - np.array(target_ee_pos))
                print(f"Position error: {error:.4f} m")
                
            except Exception as e:
                print(f"IK failed: {e}")
        
        # Capture and display camera image
        print("\\nCapturing camera image...")
        try:
            image = robot_sim.get_camera_image()
            print(f"Captured image shape: {image.shape}")
        except Exception as e:
            print(f"Camera capture failed: {e}")
        
        # Keep simulation running for observation
        print("\\nSimulation running... Press Ctrl+C to exit")
        
        # Run simulation loop
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            robot_sim.step_simulation()
            time.sleep(0.01)
            
            # Occasionally move the robot
            if int(time.time() - start_time) % 5 == 0:
                # Move to a random position within joint limits
                if robot_sim.joint_limits:
                    random_positions = []
                    for lower, upper in robot_sim.joint_limits:
                        if lower == -np.inf or upper == np.inf:
                            random_positions.append(np.random.uniform(-1, 1))
                        else:
                            random_positions.append(np.random.uniform(lower, upper))
                    
                    robot_sim.move_to_joint_positions(random_positions)
                    time.sleep(0.1)  # Small delay to avoid rapid updates
    
    except KeyboardInterrupt:
        print("\\nKeyboard interrupt received!")
    
    finally:
        # Cleanup
        robot_sim.disconnect()
        print("Simulation ended.")


if __name__ == "__main__":
    main()