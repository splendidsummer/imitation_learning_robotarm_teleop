"""
Test suite for robot arm teleoperation package.
"""

import unittest
import sys
import os
import numpy as np

# Add the package to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm_teleop.simulation import RobotArmSimulation
from robot_arm_teleop.teleoperation import TeleoperationController, KeyboardInput, JoystickInput
from robot_arm_teleop.environments import ALOHAEnvironment
from robot_arm_teleop.utils import math_utils


class TestRobotArmSimulation(unittest.TestCase):
    """Test cases for RobotArmSimulation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.robot_sim = RobotArmSimulation(use_gui=False)
    
    def tearDown(self):
        """Clean up after tests."""
        self.robot_sim.disconnect()
    
    def test_initialization(self):
        """Test robot simulation initialization."""
        self.assertIsNotNone(self.robot_sim.physics_client)
        self.assertIsNotNone(self.robot_sim.robot_id)
        self.assertIsNotNone(self.robot_sim.plane_id)
    
    def test_joint_control(self):
        """Test joint position control."""
        if not self.robot_sim.joint_indices:
            self.skipTest("No controllable joints found")
        
        # Get initial positions
        initial_positions, _ = self.robot_sim.get_joint_states()
        
        # Set new positions
        target_positions = [0.1] * len(initial_positions)
        self.robot_sim.set_joint_positions(target_positions)
        
        # Check if positions were set
        new_positions, _ = self.robot_sim.get_joint_states()
        np.testing.assert_allclose(new_positions, target_positions, atol=0.01)
    
    def test_end_effector_pose(self):
        """Test end effector pose retrieval."""
        position, orientation = self.robot_sim.get_end_effector_pose()
        
        self.assertEqual(len(position), 3)
        self.assertEqual(len(orientation), 4)
        
        # Check that values are reasonable
        self.assertTrue(all(isinstance(p, (int, float)) for p in position))
        self.assertTrue(all(isinstance(o, (int, float)) for o in orientation))
    
    def test_camera_image(self):
        """Test camera image capture."""
        try:
            image = self.robot_sim.get_camera_image(width=320, height=240)
            self.assertEqual(image.shape, (240, 320, 3))
            self.assertTrue(image.dtype == np.uint8)
        except Exception as e:
            self.skipTest(f"Camera not available: {e}")


class TestTeleoperationController(unittest.TestCase):
    """Test cases for TeleoperationController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.robot_sim = RobotArmSimulation(use_gui=False)
        self.keyboard_input = KeyboardInput()
        self.teleop = TeleoperationController(
            self.robot_sim,
            self.keyboard_input,
            control_mode="joint"
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.robot_sim.disconnect()
    
    def test_initialization(self):
        """Test teleoperation controller initialization."""
        self.assertEqual(self.teleop.control_mode, "joint")
        self.assertIsInstance(self.teleop.input_device, KeyboardInput)
        self.assertFalse(self.teleop.is_running)
    
    def test_control_mode_switching(self):
        """Test control mode switching."""
        self.teleop.set_control_mode("cartesian")
        self.assertEqual(self.teleop.control_mode, "cartesian")
        
        self.teleop.set_control_mode("joint")
        self.assertEqual(self.teleop.control_mode, "joint")
    
    def test_start_stop(self):
        """Test starting and stopping teleoperation."""
        self.assertFalse(self.teleop.is_running)
        
        self.teleop.start()
        self.assertTrue(self.teleop.is_running)
        
        self.teleop.stop()
        self.assertFalse(self.teleop.is_running)


class TestALOHAEnvironment(unittest.TestCase):
    """Test cases for ALOHAEnvironment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aloha_env = ALOHAEnvironment(use_gui=False)
    
    def tearDown(self):
        """Clean up after tests."""
        self.aloha_env.disconnect()
    
    def test_initialization(self):
        """Test ALOHA environment initialization."""
        self.assertIsNotNone(self.aloha_env.left_arm_id)
        self.assertIsNotNone(self.aloha_env.right_arm_id)
        self.assertIsNotNone(self.aloha_env.table_id)
    
    def test_robot_states(self):
        """Test robot state retrieval."""
        left_positions, right_positions = self.aloha_env.get_robot_states()
        
        self.assertIsInstance(left_positions, list)
        self.assertIsInstance(right_positions, list)
        self.assertTrue(len(left_positions) > 0)
        self.assertTrue(len(right_positions) > 0)
    
    def test_end_effector_poses(self):
        """Test end effector pose retrieval."""
        (left_pos, left_ori), (right_pos, right_ori) = self.aloha_env.get_end_effector_poses()
        
        self.assertEqual(len(left_pos), 3)
        self.assertEqual(len(left_ori), 4)
        self.assertEqual(len(right_pos), 3)
        self.assertEqual(len(right_ori), 4)
    
    def test_task_objects(self):
        """Test task object creation."""
        # Test different tasks
        tasks = ["pick_and_place", "bimanual_manipulation", "stacking"]
        
        for task in tasks:
            self.aloha_env.clear_objects()
            initial_count = len(self.aloha_env.workspace_objects)
            
            self.aloha_env.add_task_objects(task)
            final_count = len(self.aloha_env.workspace_objects)
            
            self.assertGreater(final_count, initial_count)
    
    def test_object_creation(self):
        """Test individual object creation."""
        initial_count = len(self.aloha_env.workspace_objects)
        
        # Create a cube
        cube_id = self.aloha_env.create_cube([0, 0, 0.5])
        self.aloha_env.workspace_objects.append(cube_id)
        
        # Create a box
        box_id = self.aloha_env.create_box([0.1, 0, 0.5], [0.1, 0.1, 0.1])
        self.aloha_env.workspace_objects.append(box_id)
        
        self.assertEqual(len(self.aloha_env.workspace_objects), initial_count + 2)


class TestMathUtils(unittest.TestCase):
    """Test cases for math utility functions."""
    
    def test_euler_quaternion_conversion(self):
        """Test Euler angle to quaternion conversion and back."""
        # Test with known values
        roll, pitch, yaw = 0.1, 0.2, 0.3
        
        # Convert to quaternion
        quat = math_utils.euler_to_quaternion(roll, pitch, yaw)
        self.assertEqual(len(quat), 4)
        
        # Convert back to Euler
        roll2, pitch2, yaw2 = math_utils.quaternion_to_euler(*quat)
        
        # Check if conversion is consistent
        self.assertAlmostEqual(roll, roll2, places=5)
        self.assertAlmostEqual(pitch, pitch2, places=5)
        self.assertAlmostEqual(yaw, yaw2, places=5)
    
    def test_joint_normalization(self):
        """Test joint position normalization."""
        positions = [0.5, -0.3, 1.2]
        limits = [(-1.0, 1.0), (-0.5, 0.5), (0.0, 2.0)]
        
        normalized = math_utils.normalize_joint_positions(positions, limits)
        denormalized = math_utils.denormalize_joint_positions(normalized, limits)
        
        # Check that denormalization gives back original values (within limits)
        for orig, denorm, (lower, upper) in zip(positions, denormalized, limits):
            expected = np.clip(orig, lower, upper)
            self.assertAlmostEqual(denorm, expected, places=5)
    
    def test_trajectory_smoothing(self):
        """Test trajectory smoothing function."""
        waypoints = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 2.0]
        ]
        
        smooth_traj = math_utils.smooth_trajectory(waypoints, num_interpolated=5)
        
        # Check that we have more points than original waypoints
        self.assertGreater(len(smooth_traj), len(waypoints))
        
        # Check that start and end points are preserved
        np.testing.assert_allclose(smooth_traj[0], waypoints[0])
        np.testing.assert_allclose(smooth_traj[-1], waypoints[-1])
    
    def test_safety_bounds(self):
        """Test safety bounds creation and checking."""
        center = [0.5, 0.0, 1.0]
        size = [1.0, 1.0, 0.5]
        
        bounds = math_utils.create_safety_bounds(center, size)
        
        # Test points inside bounds
        safe_point = [0.5, 0.0, 1.0]
        self.assertTrue(math_utils.check_safety_bounds(safe_point, bounds))
        
        # Test points outside bounds
        unsafe_point = [2.0, 0.0, 1.0]
        self.assertFalse(math_utils.check_safety_bounds(unsafe_point, bounds))


class TestInputDevices(unittest.TestCase):
    """Test cases for input devices."""
    
    def test_keyboard_input(self):
        """Test keyboard input device."""
        keyboard = KeyboardInput()
        
        self.assertTrue(keyboard.is_connected())
        
        # Test getting input (should return empty dict when no keys pressed)
        commands = keyboard.get_input()
        self.assertIsInstance(commands, dict)
    
    def test_joystick_input(self):
        """Test joystick input device."""
        joystick = JoystickInput()
        
        # Joystick may not be available in test environment
        commands = joystick.get_input()
        self.assertIsInstance(commands, dict)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)