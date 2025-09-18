"""
ALOHA-style robot arm environment implementation.

ALOHA (A Low-cost Open-source Hardware Assembling) is a bimanual teleoperation
system for imitation learning. This module implements a simulated version.
"""

import numpy as np
import pybullet as p
from typing import List, Tuple, Dict, Optional
from ..simulation import RobotArmSimulation


class ALOHAEnvironment:
    """
    ALOHA-style bimanual robot environment.
    
    This environment simulates two robot arms (leader and follower) for
    teleoperation and imitation learning tasks.
    """
    
    def __init__(
        self,
        use_gui: bool = True,
        workspace_size: Tuple[float, float, float] = (1.0, 1.0, 0.5),
        table_height: float = 0.4
    ):
        """
        Initialize ALOHA environment.
        
        Args:
            use_gui: Whether to show PyBullet GUI
            workspace_size: Size of the workspace (x, y, z)
            table_height: Height of the table surface
        """
        self.use_gui = use_gui
        self.workspace_size = workspace_size
        self.table_height = table_height
        
        # Robot arms
        self.left_arm = None
        self.right_arm = None
        
        # Environment objects
        self.workspace_objects = []
        
        # Task state
        self.current_task = None
        
        self.initialize_environment()
    
    def initialize_environment(self):
        """Initialize the ALOHA environment."""
        print("Initializing ALOHA environment...")
        
        # Initialize physics simulation
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            # Set camera for bimanual view
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=0,
                cameraPitch=-30,
                cameraTargetPosition=[0.0, 0.0, 0.5]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set physics parameters
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0/240.0)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Create workspace setup
        self.create_workspace()
        
        # Initialize robot arms
        self.create_robot_arms()
        
        print("ALOHA environment initialized successfully!")
    
    def create_workspace(self):
        """Create the workspace setup including table and boundaries."""
        # Main table
        table_width, table_depth = self.workspace_size[0], self.workspace_size[1]
        table_thickness = 0.05
        
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_depth/2, table_thickness/2]
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_depth/2, table_thickness/2],
            rgbaColor=[0.8, 0.7, 0.6, 1.0]
        )
        
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0, 0, self.table_height - table_thickness/2]
        )
        
        # Add workspace boundaries (optional walls)
        self.create_workspace_boundaries()
    
    def create_workspace_boundaries(self):
        """Create workspace boundaries to keep objects contained."""
        wall_height = 0.1
        wall_thickness = 0.02
        
        # Define wall positions (left, right, front, back)
        wall_positions = [
            [-self.workspace_size[0]/2, 0, self.table_height + wall_height/2],  # Left
            [self.workspace_size[0]/2, 0, self.table_height + wall_height/2],   # Right
            [0, -self.workspace_size[1]/2, self.table_height + wall_height/2],  # Front
            [0, self.workspace_size[1]/2, self.table_height + wall_height/2]    # Back
        ]
        
        wall_orientations = [
            [wall_thickness/2, self.workspace_size[1]/2, wall_height/2],  # Left
            [wall_thickness/2, self.workspace_size[1]/2, wall_height/2],  # Right
            [self.workspace_size[0]/2, wall_thickness/2, wall_height/2],  # Front
            [self.workspace_size[0]/2, wall_thickness/2, wall_height/2]   # Back
        ]
        
        for pos, size in zip(wall_positions, wall_orientations):
            wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            wall_visual = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=size,
                rgbaColor=[0.7, 0.7, 0.7, 0.8]
            )
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=pos
            )
    
    def create_robot_arms(self):
        """Create left and right robot arms."""
        # Robot arm positions (side by side)
        left_position = [-0.3, -0.3, 0.0]
        right_position = [0.3, -0.3, 0.0]
        
        # Create custom robot arm simulation instances
        # Note: We'll use separate physics clients or manage them within the main client
        
        # For now, create simple robot arms using available URDFs
        try:
            # Try to load UR5 robots for both arms
            self.left_arm_id = p.loadURDF("ur5_robot.urdf", left_position)
            self.right_arm_id = p.loadURDF("ur5_robot.urdf", right_position)
        except:
            # Fallback to simpler robots
            self.left_arm_id = p.loadURDF("r2d2.urdf", left_position)
            self.right_arm_id = p.loadURDF("r2d2.urdf", right_position)
        
        # Analyze robot joints
        self.left_arm_joints = self._analyze_robot_joints(self.left_arm_id)
        self.right_arm_joints = self._analyze_robot_joints(self.right_arm_id)
        
        print(f"Left arm: {len(self.left_arm_joints['indices'])} joints")
        print(f"Right arm: {len(self.right_arm_joints['indices'])} joints")
        
        # Set initial poses
        self.reset_robot_poses()
    
    def _analyze_robot_joints(self, robot_id: int) -> Dict:
        """Analyze robot joints and return joint information."""
        num_joints = p.getNumJoints(robot_id)
        
        joint_info = {
            'indices': [],
            'names': [],
            'limits': [],
            'types': []
        }
        
        for i in range(num_joints):
            joint_data = p.getJointInfo(robot_id, i)
            joint_type = joint_data[2]
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                joint_info['indices'].append(i)
                joint_info['names'].append(joint_data[1].decode('utf-8'))
                joint_info['limits'].append((joint_data[8], joint_data[9]))
                joint_info['types'].append(joint_type)
        
        return joint_info
    
    def reset_robot_poses(self):
        """Reset both robot arms to default poses."""
        # Reset left arm
        for i, joint_idx in enumerate(self.left_arm_joints['indices']):
            lower, upper = self.left_arm_joints['limits'][i]
            if lower == -np.inf or upper == np.inf:
                target_pos = 0.0
            else:
                target_pos = (lower + upper) / 2.0
            p.resetJointState(self.left_arm_id, joint_idx, target_pos)
        
        # Reset right arm
        for i, joint_idx in enumerate(self.right_arm_joints['indices']):
            lower, upper = self.right_arm_joints['limits'][i]
            if lower == -np.inf or upper == np.inf:
                target_pos = 0.0
            else:
                target_pos = (lower + upper) / 2.0
            p.resetJointState(self.right_arm_id, joint_idx, target_pos)
    
    def control_left_arm(self, joint_positions: List[float]):
        """Control left arm with joint positions."""
        if len(joint_positions) != len(self.left_arm_joints['indices']):
            raise ValueError(f"Expected {len(self.left_arm_joints['indices'])} positions")
        
        p.setJointMotorControlArray(
            self.left_arm_id,
            self.left_arm_joints['indices'],
            p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[100.0] * len(joint_positions)
        )
    
    def control_right_arm(self, joint_positions: List[float]):
        """Control right arm with joint positions."""
        if len(joint_positions) != len(self.right_arm_joints['indices']):
            raise ValueError(f"Expected {len(self.right_arm_joints['indices'])} positions")
        
        p.setJointMotorControlArray(
            self.right_arm_id,
            self.right_arm_joints['indices'],
            p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[100.0] * len(joint_positions)
        )
    
    def get_robot_states(self) -> Tuple[List[float], List[float]]:
        """Get current joint states for both arms."""
        # Left arm states
        left_states = p.getJointStates(self.left_arm_id, self.left_arm_joints['indices'])
        left_positions = [state[0] for state in left_states]
        
        # Right arm states
        right_states = p.getJointStates(self.right_arm_id, self.right_arm_joints['indices'])
        right_positions = [state[0] for state in right_states]
        
        return left_positions, right_positions
    
    def get_end_effector_poses(self) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
        """Get end effector poses for both arms."""
        # Assume last link is end effector for each arm
        left_ee_link = p.getNumJoints(self.left_arm_id) - 1
        right_ee_link = p.getNumJoints(self.right_arm_id) - 1
        
        # Left arm end effector
        if left_ee_link >= 0:
            left_state = p.getLinkState(self.left_arm_id, left_ee_link)
            left_pos = list(left_state[4])
            left_ori = list(left_state[5])
        else:
            left_pos, left_ori = [0, 0, 0], [0, 0, 0, 1]
        
        # Right arm end effector
        if right_ee_link >= 0:
            right_state = p.getLinkState(self.right_arm_id, right_ee_link)
            right_pos = list(right_state[4])
            right_ori = list(right_state[5])
        else:
            right_pos, right_ori = [0, 0, 0], [0, 0, 0, 1]
        
        return (left_pos, left_ori), (right_pos, right_ori)
    
    def add_task_objects(self, task_name: str):
        """Add objects for specific task."""
        self.current_task = task_name
        
        if task_name == "pick_and_place":
            self.add_pick_and_place_objects()
        elif task_name == "bimanual_manipulation":
            self.add_bimanual_objects()
        elif task_name == "stacking":
            self.add_stacking_objects()
        else:
            print(f"Unknown task: {task_name}")
    
    def add_pick_and_place_objects(self):
        """Add objects for pick and place task."""
        # Add some cubes to pick up
        cube_positions = [
            [0.2, 0.1, self.table_height + 0.05],
            [-0.2, 0.1, self.table_height + 0.05],
            [0.0, 0.2, self.table_height + 0.05]
        ]
        
        for i, pos in enumerate(cube_positions):
            cube_id = self.create_cube(pos, size=0.03, color=[1, 0, 0, 1])
            self.workspace_objects.append(cube_id)
    
    def add_bimanual_objects(self):
        """Add objects for bimanual manipulation task."""
        # Add a larger object that requires two hands
        box_pos = [0.0, 0.1, self.table_height + 0.1]
        box_id = self.create_box(box_pos, size=[0.15, 0.05, 0.2], color=[0, 1, 0, 1])
        self.workspace_objects.append(box_id)
    
    def add_stacking_objects(self):
        """Add objects for stacking task."""
        # Add multiple blocks for stacking
        block_positions = [
            [0.1, 0.0, self.table_height + 0.025],
            [0.0, 0.1, self.table_height + 0.025],
            [-0.1, 0.0, self.table_height + 0.025]
        ]
        
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        
        for i, (pos, color) in enumerate(zip(block_positions, colors)):
            block_id = self.create_cube(pos, size=0.05, color=color)
            self.workspace_objects.append(block_id)
    
    def create_cube(self, position: List[float], size: float = 0.05, color: List[float] = None) -> int:
        """Create a cube object in the environment."""
        if color is None:
            color = [1, 0, 0, 1]
        
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2] * 3)
        cube_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[size/2] * 3,
            rgbaColor=color
        )
        
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=position
        )
        
        return cube_id
    
    def create_box(self, position: List[float], size: List[float], color: List[float] = None) -> int:
        """Create a box object in the environment."""
        if color is None:
            color = [0, 1, 0, 1]
        
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
        box_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s/2 for s in size],
            rgbaColor=color
        )
        
        box_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=position
        )
        
        return box_id
    
    def clear_objects(self):
        """Clear all task objects from the environment."""
        for obj_id in self.workspace_objects:
            p.removeBody(obj_id)
        self.workspace_objects.clear()
    
    def step_simulation(self):
        """Step the physics simulation."""
        p.stepSimulation()
    
    def get_camera_images(self, width: int = 640, height: int = 480) -> Dict[str, np.ndarray]:
        """Get camera images from multiple viewpoints."""
        images = {}
        
        # Front view
        view_matrix_front = p.computeViewMatrix(
            cameraEyePosition=[0.0, -1.0, 1.0],
            cameraTargetPosition=[0.0, 0.0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
        
        # Side view
        view_matrix_side = p.computeViewMatrix(
            cameraEyePosition=[1.0, 0.0, 1.0],
            cameraTargetPosition=[0.0, 0.0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=3.0
        )
        
        # Capture images
        for name, view_matrix in [("front", view_matrix_front), ("side", view_matrix_side)]:
            _, _, rgb_img, _, _ = p.getCameraImage(
                width, height, view_matrix, projection_matrix
            )
            rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
            images[name] = rgb_array
        
        return images
    
    def disconnect(self):
        """Disconnect from PyBullet simulation."""
        if hasattr(self, 'physics_client'):
            p.disconnect()


# Export the main class
from .aloha_environment import ALOHAEnvironment

__all__ = ["ALOHAEnvironment"]