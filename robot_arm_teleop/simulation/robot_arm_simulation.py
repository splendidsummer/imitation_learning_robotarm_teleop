"""
Core robot arm simulation using PyBullet physics engine.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any


class RobotArmSimulation:
    """
    Main simulation class for robot arm environments using PyBullet.
    
    This class handles the physics simulation, robot loading, and basic control
    for various robot arm configurations.
    """
    
    def __init__(
        self,
        robot_urdf: str = None,
        use_gui: bool = True,
        time_step: float = 1.0/240.0,
        gravity: Tuple[float, float, float] = (0, 0, -9.81)
    ):
        """
        Initialize the robot arm simulation.
        
        Args:
            robot_urdf: Path to robot URDF file. If None, uses default UR5
            use_gui: Whether to show PyBullet GUI
            time_step: Physics simulation time step
            gravity: Gravity vector (x, y, z)
        """
        self.use_gui = use_gui
        self.time_step = time_step
        self.gravity = gravity
        self.robot_urdf = robot_urdf
        
        # PyBullet physics client
        self.physics_client = None
        
        # Robot and environment objects
        self.robot_id = None
        self.plane_id = None
        self.table_id = None
        
        # Robot configuration
        self.joint_indices = []
        self.joint_names = []
        self.joint_limits = []
        self.end_effector_link = None
        
        # Control state
        self.target_joint_positions = []
        self.current_joint_positions = []
        
        self.initialize_simulation()
    
    def initialize_simulation(self):
        """Initialize PyBullet physics simulation."""
        # Connect to physics server
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            # Set camera position for better view
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0, 0.5]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set additional search path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure physics
        p.setGravity(*self.gravity)
        p.setTimeStep(self.time_step)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table
        self.load_table()
        
        # Load robot
        self.load_robot()
        
        print("Robot arm simulation initialized successfully!")
    
    def load_table(self):
        """Load a table for the robot to work on."""
        # Create a simple table using a box
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.6, 0.4, 0.02]
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.6, 0.4, 0.02],
            rgbaColor=[0.8, 0.6, 0.4, 1.0]
        )
        
        self.table_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0, 0.4]  # Position table in front of robot
        )
    
    def load_robot(self):
        """Load the robot arm URDF."""
        if self.robot_urdf is None:
            # Use default UR5 robot if no URDF specified
            try:
                self.robot_id = p.loadURDF("ur5_robot.urdf", [0, 0, 0])
            except:
                # Fallback to a simple robot if UR5 not available
                self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0])
        else:
            self.robot_id = p.loadURDF(self.robot_urdf, [0, 0, 0])
        
        # Get joint information
        self._analyze_robot_joints()
        
        # Initialize joint positions
        self.reset_robot_pose()
    
    def _analyze_robot_joints(self):
        """Analyze robot joints and extract relevant information."""
        num_joints = p.getNumJoints(self.robot_id)
        
        self.joint_indices = []
        self.joint_names = []
        self.joint_limits = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Only consider revolute and prismatic joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
                
                # Get joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))
        
        print(f"Found {len(self.joint_indices)} controllable joints:")
        for i, (idx, name) in enumerate(zip(self.joint_indices, self.joint_names)):
            print(f"  Joint {idx}: {name} - Limits: {self.joint_limits[i]}")
        
        # Set end effector link (assume last link)
        if num_joints > 0:
            self.end_effector_link = num_joints - 1
    
    def reset_robot_pose(self):
        """Reset robot to a default pose."""
        if not self.joint_indices:
            return
        
        # Set to middle position of joint ranges
        target_positions = []
        for lower, upper in self.joint_limits:
            if lower == -np.inf or upper == np.inf:
                target_positions.append(0.0)
            else:
                target_positions.append((lower + upper) / 2.0)
        
        self.set_joint_positions(target_positions)
        self.target_joint_positions = target_positions.copy()
    
    def set_joint_positions(self, positions: List[float]):
        """Set robot joint positions directly."""
        if len(positions) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} positions, got {len(positions)}")
        
        for i, (joint_idx, pos) in enumerate(zip(self.joint_indices, positions)):
            p.resetJointState(self.robot_id, joint_idx, pos)
        
        self.current_joint_positions = positions.copy()
    
    def move_to_joint_positions(self, positions: List[float], max_force: float = 100.0):
        """Move robot to target joint positions using position control."""
        if len(positions) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} positions, got {len(positions)}")
        
        self.target_joint_positions = positions.copy()
        
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=positions,
            forces=[max_force] * len(self.joint_indices)
        )
    
    def get_joint_states(self) -> Tuple[List[float], List[float]]:
        """Get current joint positions and velocities."""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        
        self.current_joint_positions = positions
        return positions, velocities
    
    def get_end_effector_pose(self) -> Tuple[List[float], List[float]]:
        """Get end effector position and orientation."""
        if self.end_effector_link is None:
            return [0, 0, 0], [0, 0, 0, 1]
        
        link_state = p.getLinkState(self.robot_id, self.end_effector_link)
        position = list(link_state[4])  # World position
        orientation = list(link_state[5])  # World orientation (quaternion)
        
        return position, orientation
    
    def inverse_kinematics(self, target_position: List[float], target_orientation: List[float] = None) -> List[float]:
        """
        Compute inverse kinematics for target end effector pose.
        
        Args:
            target_position: Target [x, y, z] position
            target_orientation: Target orientation as quaternion [x, y, z, w]
            
        Returns:
            Joint positions to reach target pose
        """
        if self.end_effector_link is None:
            raise ValueError("End effector link not defined")
        
        if target_orientation is None:
            # Compute IK with position only
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                target_position
            )
        else:
            # Compute IK with position and orientation
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                target_position,
                target_orientation
            )
        
        # Filter to only controlled joints
        return [joint_poses[i] for i in self.joint_indices]
    
    def step_simulation(self):
        """Step the physics simulation forward."""
        p.stepSimulation()
    
    def get_camera_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Capture camera image from simulation."""
        # Set up camera
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.0, 1.0, 1.0],
            cameraTargetPosition=[0.5, 0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=3.0
        )
        
        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width, height, view_matrix, projection_matrix
        )
        
        # Convert to numpy array and remove alpha channel
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        
        return rgb_array
    
    def add_object(self, urdf_path: str, position: List[float], orientation: List[float] = None) -> int:
        """Add an object to the simulation."""
        if orientation is None:
            orientation = [0, 0, 0, 1]  # No rotation
        
        object_id = p.loadURDF(urdf_path, position, orientation)
        return object_id
    
    def disconnect(self):
        """Disconnect from PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.disconnect()


# Export the main class
from .robot_arm_simulation import RobotArmSimulation

__all__ = ["RobotArmSimulation"]