#!/usr/bin/env python3
"""
Keyboard teleoperation example.

This script demonstrates keyboard-based teleoperation of a robot arm.
Use WASD keys and others to control the robot joints or end effector.
"""

import sys
import os
import time

# Add the package to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm_teleop.simulation import RobotArmSimulation
from robot_arm_teleop.teleoperation import TeleoperationController, KeyboardInput


def main():
    """Run keyboard teleoperation example."""
    print("Starting keyboard teleoperation example...")
    
    # Create robot arm simulation
    robot_sim = RobotArmSimulation(use_gui=True)
    
    # Create keyboard input device
    keyboard_input = KeyboardInput()
    
    # Create teleoperation controller
    teleop_controller = TeleoperationController(
        robot_simulation=robot_sim,
        input_device=keyboard_input,
        control_mode="joint",  # Start with joint control
        update_rate=60.0
    )
    
    try:
        print("\\n" + "="*50)
        print("KEYBOARD TELEOPERATION CONTROLS")
        print("="*50)
        print("Joint Control (Current Mode):")
        print("  Q/W: Joint 0    A/S: Joint 1    Z/X: Joint 2")
        print("  E/R: Joint 3    D/F: Joint 4    C/V: Joint 5")
        print("")
        print("End Effector Control:")
        print("  Arrow Keys: X/Y movement")
        print("  Page Up/Down: Z movement")
        print("")
        print("Commands:")
        print("  SPACE: Pause/Resume")
        print("  H: Go to home position")
        print("  J: Switch to joint control mode")
        print("  K: Switch to Cartesian control mode")
        print("  ESC: Exit")
        print("="*50)
        
        # Add mode switching functionality
        def check_mode_switch():
            import pygame
            keys = pygame.key.get_pressed()
            if keys[pygame.K_j]:
                teleop_controller.set_control_mode("joint")
                time.sleep(0.2)  # Debounce
            elif keys[pygame.K_k]:
                teleop_controller.set_control_mode("cartesian")
                time.sleep(0.2)  # Debounce
        
        # Start teleoperation
        teleop_controller.start()
        
        # Main control loop
        print("\\nTeleoperation started! Use the controls above to move the robot.")
        
        last_status_time = time.time()
        
        while teleop_controller.is_running:
            # Update teleoperation
            teleop_controller.update()
            
            # Check for mode switching
            check_mode_switch()
            
            # Step simulation
            robot_sim.step_simulation()
            
            # Print status occasionally
            current_time = time.time()
            if current_time - last_status_time > 5.0:
                positions, velocities = robot_sim.get_joint_states()
                ee_pos, ee_ori = robot_sim.get_end_effector_pose()
                
                print(f"\\n[Status] Mode: {teleop_controller.control_mode}")
                print(f"Joint positions: {[f'{p:.3f}' for p in positions]}")
                print(f"End effector pos: {[f'{p:.3f}' for p in ee_pos]}")
                
                last_status_time = current_time
            
            # Small delay to maintain update rate
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\\nKeyboard interrupt received!")
    
    except Exception as e:
        print(f"\\nError occurred: {e}")
    
    finally:
        # Cleanup
        teleop_controller.stop()
        robot_sim.disconnect()
        print("Teleoperation ended.")


if __name__ == "__main__":
    main()