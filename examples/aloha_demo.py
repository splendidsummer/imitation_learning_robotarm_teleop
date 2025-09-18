#!/usr/bin/env python3
"""
ALOHA environment example.

This script demonstrates the ALOHA bimanual robot environment
with various tasks and teleoperation capabilities.
"""

import sys
import os
import time
import numpy as np

# Add the package to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm_teleop.environments import ALOHAEnvironment


def main():
    """Run ALOHA environment example."""
    print("Starting ALOHA environment example...")
    
    # Create ALOHA environment
    aloha_env = ALOHAEnvironment(use_gui=True)
    
    try:
        # Display initial information
        print("\\nALOHA Environment initialized!")
        print(f"Left arm joints: {len(aloha_env.left_arm_joints['indices'])}")
        print(f"Right arm joints: {len(aloha_env.right_arm_joints['indices'])}")
        
        # Get initial robot states
        left_positions, right_positions = aloha_env.get_robot_states()
        print(f"\\nInitial left arm positions: {[f'{p:.3f}' for p in left_positions]}")
        print(f"Initial right arm positions: {[f'{p:.3f}' for p in right_positions]}")
        
        # Get initial end effector poses
        (left_pos, left_ori), (right_pos, right_ori) = aloha_env.get_end_effector_poses()
        print(f"\\nLeft end effector: {[f'{p:.3f}' for p in left_pos]}")
        print(f"Right end effector: {[f'{p:.3f}' for p in right_pos]}")
        
        # Demonstrate different tasks
        tasks = ["pick_and_place", "bimanual_manipulation", "stacking"]
        
        for task in tasks:
            print(f"\\n{'='*50}")
            print(f"DEMONSTRATING TASK: {task.upper()}")
            print(f"{'='*50}")
            
            # Clear previous objects and add new ones
            aloha_env.clear_objects()
            aloha_env.add_task_objects(task)
            
            # Let objects settle
            for _ in range(50):
                aloha_env.step_simulation()
                time.sleep(0.02)
            
            # Demonstrate coordinated arm movements
            if task == "pick_and_place":
                demonstrate_pick_and_place(aloha_env)
            elif task == "bimanual_manipulation":
                demonstrate_bimanual_manipulation(aloha_env)
            elif task == "stacking":
                demonstrate_stacking(aloha_env)
            
            # Pause between tasks
            print(f"Task {task} demonstration complete. Waiting 3 seconds...")
            time.sleep(3)
        
        # Capture camera images
        print("\\nCapturing camera images...")
        try:
            images = aloha_env.get_camera_images()
            for view_name, image in images.items():
                print(f"{view_name} camera image shape: {image.shape}")
        except Exception as e:
            print(f"Camera capture failed: {e}")
        
        # Free movement demonstration
        print("\\nDemonstrating free movement...")
        demonstrate_free_movement(aloha_env)
        
        print("\\nDemo complete. Simulation will continue for observation...")
        
        # Keep simulation running for observation
        start_time = time.time()
        while time.time() - start_time < 20:  # Run for 20 more seconds
            aloha_env.step_simulation()
            time.sleep(0.02)
    
    except KeyboardInterrupt:
        print("\\nKeyboard interrupt received!")
    
    finally:
        # Cleanup
        aloha_env.disconnect()
        print("ALOHA environment demo ended.")


def demonstrate_pick_and_place(aloha_env):
    """Demonstrate pick and place movements."""
    print("Moving arms to pick up objects...")
    
    # Get current positions
    left_positions, right_positions = aloha_env.get_robot_states()
    
    # Create simple movement sequence
    movements = [
        # Move both arms up slightly
        (add_joint_offset(left_positions, [0, 0, 0.2, 0, 0, 0]),
         add_joint_offset(right_positions, [0, 0, 0.2, 0, 0, 0])),
        
        # Move arms to different positions
        (add_joint_offset(left_positions, [0.3, -0.2, 0.1, 0, 0, 0]),
         add_joint_offset(right_positions, [-0.3, -0.2, 0.1, 0, 0, 0])),
        
        # Return to center
        (left_positions, right_positions)
    ]
    
    execute_movement_sequence(aloha_env, movements)


def demonstrate_bimanual_manipulation(aloha_env):
    """Demonstrate bimanual manipulation movements."""
    print("Demonstrating bimanual coordination...")
    
    # Get current positions
    left_positions, right_positions = aloha_env.get_robot_states()
    
    # Create coordinated movement sequence
    movements = [
        # Both arms move inward
        (add_joint_offset(left_positions, [0.2, 0, 0, 0, 0, 0]),
         add_joint_offset(right_positions, [-0.2, 0, 0, 0, 0, 0])),
        
        # Both arms move up together
        (add_joint_offset(left_positions, [0.2, 0, 0.3, 0, 0, 0]),
         add_joint_offset(right_positions, [-0.2, 0, 0.3, 0, 0, 0])),
        
        # Return to start
        (left_positions, right_positions)
    ]
    
    execute_movement_sequence(aloha_env, movements)


def demonstrate_stacking(aloha_env):
    """Demonstrate stacking movements."""
    print("Demonstrating stacking motions...")
    
    # Get current positions
    left_positions, right_positions = aloha_env.get_robot_states()
    
    # Create stacking movement sequence
    movements = [
        # Left arm picks up, right arm supports
        (add_joint_offset(left_positions, [0.1, -0.1, 0.2, 0, 0, 0]),
         add_joint_offset(right_positions, [0, -0.1, 0.1, 0, 0, 0])),
        
        # Left arm places while right arm stabilizes
        (add_joint_offset(left_positions, [0.1, 0.1, 0.3, 0, 0, 0]),
         add_joint_offset(right_positions, [0.1, 0.1, 0.1, 0, 0, 0])),
        
        # Both arms return
        (left_positions, right_positions)
    ]
    
    execute_movement_sequence(aloha_env, movements)


def demonstrate_free_movement(aloha_env):
    """Demonstrate free movement patterns."""
    print("Demonstrating free movement patterns...")
    
    # Get current positions
    left_positions, right_positions = aloha_env.get_robot_states()
    
    # Create smooth movement patterns
    num_steps = 60
    for i in range(num_steps):
        t = i / num_steps * 2 * np.pi
        
        # Sinusoidal movements
        left_offset = [0.1 * np.sin(t), 0.1 * np.cos(t), 0.05 * np.sin(2*t), 0, 0, 0]
        right_offset = [0.1 * np.sin(t + np.pi), 0.1 * np.cos(t + np.pi), 0.05 * np.sin(2*t + np.pi), 0, 0, 0]
        
        left_target = add_joint_offset(left_positions, left_offset)
        right_target = add_joint_offset(right_positions, right_offset)
        
        # Apply movements with safety limits
        left_target = apply_joint_limits(left_target, aloha_env.left_arm_joints['limits'])
        right_target = apply_joint_limits(right_target, aloha_env.right_arm_joints['limits'])
        
        aloha_env.control_left_arm(left_target)
        aloha_env.control_right_arm(right_target)
        
        # Step simulation
        for _ in range(3):
            aloha_env.step_simulation()
            time.sleep(0.01)


def add_joint_offset(positions, offsets):
    """Add offset to joint positions safely."""
    result = positions.copy()
    for i in range(min(len(result), len(offsets))):
        result[i] += offsets[i]
    return result


def apply_joint_limits(positions, limits):
    """Apply joint limits to positions."""
    result = positions.copy()
    for i in range(min(len(result), len(limits))):
        lower, upper = limits[i]
        if lower != -np.inf and upper != np.inf:
            result[i] = np.clip(result[i], lower, upper)
    return result


def execute_movement_sequence(aloha_env, movements):
    """Execute a sequence of movements."""
    for left_target, right_target in movements:
        # Apply joint limits
        left_target = apply_joint_limits(left_target, aloha_env.left_arm_joints['limits'])
        right_target = apply_joint_limits(right_target, aloha_env.right_arm_joints['limits'])
        
        # Send commands
        aloha_env.control_left_arm(left_target)
        aloha_env.control_right_arm(right_target)
        
        # Wait for movement to complete
        for _ in range(50):
            aloha_env.step_simulation()
            time.sleep(0.02)


if __name__ == "__main__":
    main()