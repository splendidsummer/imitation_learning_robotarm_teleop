#!/usr/bin/env python3
"""
Robot Arm Simulation Demo

This script demonstrates the robot arm simulation environment with various examples.
"""

import sys
import os
import numpy as np

# Add the package to path for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm_sim import RobotArm, RobotArmSimulation


def demo_basic_kinematics():
    """Demonstrate basic forward and inverse kinematics."""
    print("=== Basic Kinematics Demo ===")
    
    # Create robot arm
    robot = RobotArm(link_lengths=[1.0, 0.8, 0.6])
    
    # Test forward kinematics
    print("\\nForward Kinematics Test:")
    robot.set_joint_angles([np.pi/4, np.pi/6, -np.pi/8])
    end_pos, joint_positions = robot.forward_kinematics()
    
    angles_deg = np.degrees(robot.get_joint_angles())
    print(f"Joint angles: [{angles_deg[0]:.1f}, {angles_deg[1]:.1f}, {angles_deg[2]:.1f}] degrees")
    print(f"End effector position: ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
    
    # Test inverse kinematics
    print("\\nInverse Kinematics Test:")
    target_x, target_y = 1.2, 0.8
    print(f"Target position: ({target_x}, {target_y})")
    
    success = robot.move_to_position(target_x, target_y)
    if success:
        print("✓ Target reached!")
        end_pos, _ = robot.forward_kinematics()
        print(f"Actual position: ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
        angles_deg = np.degrees(robot.get_joint_angles())
        print(f"Joint angles: [{angles_deg[0]:.1f}, {angles_deg[1]:.1f}, {angles_deg[2]:.1f}] degrees")
    else:
        print("✗ Target unreachable")


def demo_interactive_simulation():
    """Run interactive simulation with mouse control."""
    print("\\n=== Interactive Simulation Demo ===")
    print("Instructions:")
    print("- Left click to set target position")
    print("- Press 'R' key to reset robot")
    print("- Press 'C' key to clear trajectory")
    print("- Close window to exit")
    
    # Create simulation
    robot = RobotArm()
    sim = RobotArmSimulation(robot)
    
    # Run interactive mode
    sim.run_interactive()


def demo_predefined_movement():
    """Demonstrate robot movement with predefined trajectory."""
    print("\\n=== Predefined Movement Demo ===")
    
    # Create simulation
    robot = RobotArm()
    sim = RobotArmSimulation(robot)
    
    # Run demonstration
    sim.demonstrate_movement()


def demo_custom_control():
    """Demonstrate custom control callback."""
    print("\\n=== Custom Control Demo ===")
    
    robot = RobotArm()
    sim = RobotArmSimulation(robot)
    
    def circle_control(simulation):
        """Move end effector in a circle."""
        t = simulation.time
        radius = 1.0
        center_x, center_y = 0.5, 0.5
        
        target_x = center_x + radius * np.cos(t)
        target_y = center_y + radius * np.sin(t)
        
        simulation.set_target_position(target_x, target_y)
        simulation.move_to_target()
    
    sim.set_control_callback(circle_control)
    sim.run_animation(duration=8.0)


def demo_workspace_analysis():
    """Analyze and visualize robot workspace."""
    print("\\n=== Workspace Analysis ===")
    
    robot = RobotArm()
    
    # Get workspace bounds
    min_x, max_x, min_y, max_y = robot.get_workspace_bounds()
    print(f"Workspace bounds:")
    print(f"  X: [{min_x:.2f}, {max_x:.2f}]")
    print(f"  Y: [{min_y:.2f}, {max_y:.2f}]")
    
    # Test reachability at various points
    test_points = [
        (1.0, 0.5),   # Should be reachable
        (2.0, 1.0),   # Might be unreachable
        (0.5, 1.5),   # Should be reachable
        (2.5, 2.5),   # Should be unreachable
    ]
    
    print("\\nReachability test:")
    for x, y in test_points:
        angles = robot.inverse_kinematics_2d(x, y)
        if angles is not None:
            print(f"  ({x:4.1f}, {y:4.1f}): ✓ Reachable")
        else:
            print(f"  ({x:4.1f}, {y:4.1f}): ✗ Unreachable")


def print_menu():
    """Print demo menu."""
    print("\\n" + "="*50)
    print("ROBOT ARM SIMULATION DEMO")
    print("="*50)
    print("1. Basic Kinematics Demo")
    print("2. Interactive Simulation (requires display)")
    print("3. Predefined Movement Demo (requires display)")
    print("4. Custom Control Demo (requires display)")
    print("5. Workspace Analysis")
    print("6. Run All Demos")
    print("0. Exit")
    print("="*50)


def main():
    """Main demo function."""
    try:
        import matplotlib.pyplot as plt
        has_display = True
    except Exception:
        print("Warning: Display not available. Some demos will be skipped.")
        has_display = False
    
    while True:
        print_menu()
        
        try:
            choice = input("\\nSelect demo (0-6): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                demo_basic_kinematics()
            elif choice == '2':
                if has_display:
                    demo_interactive_simulation()
                else:
                    print("Display not available for interactive demo.")
            elif choice == '3':
                if has_display:
                    demo_predefined_movement()
                else:
                    print("Display not available for movement demo.")
            elif choice == '4':
                if has_display:
                    demo_custom_control()
                else:
                    print("Display not available for custom control demo.")
            elif choice == '5':
                demo_workspace_analysis()
            elif choice == '6':
                demo_basic_kinematics()
                demo_workspace_analysis()
                if has_display:
                    print("\\nRunning visual demos...")
                    demo_predefined_movement()
                    demo_custom_control()
                else:
                    print("Skipping visual demos (no display).")
            else:
                print("Invalid choice. Please select 0-6.")
                
        except KeyboardInterrupt:
            print("\\n\\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error running demo: {e}")
        
        input("\\nPress Enter to continue...")


if __name__ == "__main__":
    main()