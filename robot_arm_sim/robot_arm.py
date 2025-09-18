"""
Robot Arm implementation with forward kinematics and basic control.
"""

import numpy as np
from typing import List, Tuple, Optional


class RobotArm:
    """
    A simple 3-DOF robot arm simulation with revolute joints.
    
    The robot arm consists of:
    - Base (fixed)
    - 3 revolute joints
    - End effector
    """
    
    def __init__(self, link_lengths: List[float] = None):
        """
        Initialize the robot arm.
        
        Args:
            link_lengths: List of link lengths [L1, L2, L3]. Defaults to [1.0, 0.8, 0.6]
        """
        if link_lengths is None:
            link_lengths = [1.0, 0.8, 0.6]
        
        self.link_lengths = np.array(link_lengths)
        self.num_joints = len(link_lengths)
        
        # Joint angles (in radians)
        self.joint_angles = np.zeros(self.num_joints)
        
        # Joint limits (in radians)
        self.joint_limits = [
            (-np.pi, np.pi),  # Joint 1: full rotation
            (-np.pi/2, np.pi/2),  # Joint 2: limited range
            (-np.pi/2, np.pi/2)   # Joint 3: limited range
        ]
        
        # Joint velocities
        self.joint_velocities = np.zeros(self.num_joints)
        
    def set_joint_angles(self, angles: List[float]) -> None:
        """
        Set joint angles with limit checking.
        
        Args:
            angles: List of joint angles in radians
        """
        if len(angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} angles, got {len(angles)}")
        
        # Apply joint limits
        for i, angle in enumerate(angles):
            min_limit, max_limit = self.joint_limits[i]
            self.joint_angles[i] = np.clip(angle, min_limit, max_limit)
    
    def get_joint_angles(self) -> np.ndarray:
        """Get current joint angles."""
        return self.joint_angles.copy()
    
    def forward_kinematics(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute forward kinematics to get end effector position and all joint positions.
        
        Returns:
            Tuple of (end_effector_position, joint_positions)
        """
        # Start at base (origin)
        positions = [np.array([0.0, 0.0])]
        current_pos = np.array([0.0, 0.0])
        current_angle = 0.0
        
        # Compute position of each joint
        for i in range(self.num_joints):
            current_angle += self.joint_angles[i]
            
            # Compute next joint position
            dx = self.link_lengths[i] * np.cos(current_angle)
            dy = self.link_lengths[i] * np.sin(current_angle)
            
            current_pos = current_pos + np.array([dx, dy])
            positions.append(current_pos.copy())
        
        return positions[-1], positions
    
    def inverse_kinematics_2d(self, target_x: float, target_y: float) -> Optional[np.ndarray]:
        """
        Simple 2-DOF inverse kinematics for the first two joints.
        
        Args:
            target_x: Target x position
            target_y: Target y position
            
        Returns:
            Joint angles if solution exists, None otherwise
        """
        # Use first two links for 2-DOF IK
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # Distance to target
        distance = np.sqrt(target_x**2 + target_y**2)
        
        # Check if target is reachable
        if distance > (L1 + L2) or distance < abs(L1 - L2):
            return None
        
        # Cosine rule for second joint
        cos_q2 = (distance**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_q2 = np.clip(cos_q2, -1, 1)  # Numerical stability
        
        # Two solutions for elbow up/down
        q2 = np.arccos(cos_q2)  # Elbow up configuration
        
        # First joint angle
        beta = np.arctan2(target_y, target_x)
        alpha = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q1 = beta - alpha
        
        # Keep third joint at current angle for now
        q3 = self.joint_angles[2]
        
        return np.array([q1, q2, q3])
    
    def move_to_position(self, target_x: float, target_y: float) -> bool:
        """
        Move the robot arm to a target position using inverse kinematics.
        
        Args:
            target_x: Target x position
            target_y: Target y position
            
        Returns:
            True if successful, False if target unreachable
        """
        joint_angles = self.inverse_kinematics_2d(target_x, target_y)
        
        if joint_angles is not None:
            self.set_joint_angles(joint_angles)
            return True
        
        return False
    
    def get_workspace_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get approximate workspace bounds (min_x, max_x, min_y, max_y).
        """
        max_reach = np.sum(self.link_lengths)
        min_reach = abs(self.link_lengths[0] - np.sum(self.link_lengths[1:]))
        
        return -max_reach, max_reach, -max_reach, max_reach
    
    def get_link_positions(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get start and end positions of each link for visualization.
        
        Returns:
            List of (start_pos, end_pos) tuples for each link
        """
        _, joint_positions = self.forward_kinematics()
        
        links = []
        for i in range(len(joint_positions) - 1):
            links.append((joint_positions[i], joint_positions[i + 1]))
        
        return links