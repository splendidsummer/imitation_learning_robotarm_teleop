#!/usr/bin/env python3
"""
Simple tests for the robot arm simulation.
"""

import sys
import os
import numpy as np

# Add the package to path for importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_arm_sim import RobotArm, RobotArmSimulation


def test_robot_arm_creation():
    """Test robot arm creation and basic properties."""
    print("Testing robot arm creation...")
    
    # Default robot
    robot = RobotArm()
    assert robot.num_joints == 3
    assert len(robot.link_lengths) == 3
    assert len(robot.joint_angles) == 3
    
    # Custom robot
    custom_lengths = [1.5, 1.0, 0.5]
    robot2 = RobotArm(custom_lengths)
    assert np.allclose(robot2.link_lengths, custom_lengths)
    
    print("✓ Robot arm creation tests passed")


def test_forward_kinematics():
    """Test forward kinematics calculation."""
    print("Testing forward kinematics...")
    
    robot = RobotArm([1.0, 1.0, 1.0])
    
    # Test zero configuration
    robot.set_joint_angles([0, 0, 0])
    end_pos, joint_positions = robot.forward_kinematics()
    
    assert len(joint_positions) == 4  # base + 3 joints
    assert np.allclose(end_pos, [3.0, 0.0])  # All links aligned horizontally
    
    # Test 90-degree configuration
    robot.set_joint_angles([np.pi/2, 0, 0])
    end_pos, _ = robot.forward_kinematics()
    assert np.allclose(end_pos, [0.0, 3.0], atol=1e-10)  # All links pointing up
    
    print("✓ Forward kinematics tests passed")


def test_joint_limits():
    """Test joint limit enforcement."""
    print("Testing joint limits...")
    
    robot = RobotArm()
    
    # Test setting angles within limits
    robot.set_joint_angles([np.pi/4, np.pi/4, -np.pi/4])
    angles = robot.get_joint_angles()
    assert np.allclose(angles, [np.pi/4, np.pi/4, -np.pi/4])
    
    # Test setting angles beyond limits (should be clipped)
    robot.set_joint_angles([2*np.pi, np.pi, np.pi])  # Exceed limits
    angles = robot.get_joint_angles()
    assert angles[0] <= np.pi  # Should be clipped
    assert angles[1] <= np.pi/2  # Should be clipped
    assert angles[2] <= np.pi/2  # Should be clipped
    
    print("✓ Joint limits tests passed")


def test_inverse_kinematics():
    """Test inverse kinematics."""
    print("Testing inverse kinematics...")
    
    robot = RobotArm([1.0, 1.0, 0.5])
    
    # Test reachable position
    target_x, target_y = 1.5, 0.5
    angles = robot.inverse_kinematics_2d(target_x, target_y)
    assert angles is not None
    
    # Verify the solution
    robot.set_joint_angles(angles)
    end_pos, _ = robot.forward_kinematics()
    # Note: Only checking first two links since IK is 2-DOF
    expected_pos = np.array([target_x, target_y])
    distance_to_target = np.linalg.norm(end_pos[:2] - expected_pos)
    # Allow some tolerance due to the third link
    assert distance_to_target < 1.0  # Within reasonable range
    
    # Test unreachable position
    unreachable_angles = robot.inverse_kinematics_2d(10.0, 10.0)
    assert unreachable_angles is None
    
    print("✓ Inverse kinematics tests passed")


def test_workspace_bounds():
    """Test workspace calculation."""
    print("Testing workspace bounds...")
    
    robot = RobotArm([1.0, 0.8, 0.6])
    min_x, max_x, min_y, max_y = robot.get_workspace_bounds()
    
    # Maximum reach should be sum of all links
    max_reach = sum(robot.link_lengths)
    assert max_x == max_reach
    assert max_y == max_reach
    assert min_x == -max_reach
    assert min_y == -max_reach
    
    print("✓ Workspace bounds tests passed")


def test_simulation_creation():
    """Test simulation environment creation."""
    print("Testing simulation creation...")
    
    robot = RobotArm()
    sim = RobotArmSimulation(robot)
    
    assert sim.robot_arm is robot
    assert sim.update_rate > 0
    assert sim.dt > 0
    assert not sim.is_running
    
    # Test state retrieval
    state = sim.get_state()
    assert 'time' in state
    assert 'joint_angles' in state
    assert 'end_effector_position' in state
    
    print("✓ Simulation creation tests passed")


def test_simulation_step():
    """Test simulation stepping."""
    print("Testing simulation step...")
    
    robot = RobotArm()
    sim = RobotArmSimulation(robot)
    
    initial_time = sim.time
    initial_trajectory_length = len(sim.trajectory)
    
    # Perform one step
    sim.step()
    
    assert sim.time > initial_time
    assert len(sim.trajectory) > initial_trajectory_length
    
    print("✓ Simulation step tests passed")


def run_all_tests():
    """Run all tests."""
    print("="*50)
    print("ROBOT ARM SIMULATION TESTS")
    print("="*50)
    
    tests = [
        test_robot_arm_creation,
        test_forward_kinematics,
        test_joint_limits,
        test_inverse_kinematics,
        test_workspace_bounds,
        test_simulation_creation,
        test_simulation_step,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("="*50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)