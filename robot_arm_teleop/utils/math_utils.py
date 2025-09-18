"""
Utility functions for robot arm teleoperation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import cv2


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
        
    Returns:
        Quaternion as [x, y, z, w]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def normalize_joint_positions(positions: List[float], limits: List[Tuple[float, float]]) -> List[float]:
    """
    Normalize joint positions to [-1, 1] range based on joint limits.
    
    Args:
        positions: Joint positions
        limits: Joint limits as [(min, max), ...]
        
    Returns:
        Normalized positions in [-1, 1] range
    """
    normalized = []
    for pos, (min_val, max_val) in zip(positions, limits):
        if min_val == -np.inf or max_val == np.inf:
            # For joints without limits, assume they're already normalized
            normalized.append(np.clip(pos, -1, 1))
        else:
            # Normalize to [-1, 1]
            range_size = max_val - min_val
            center = (max_val + min_val) / 2
            norm_pos = 2 * (pos - center) / range_size
            normalized.append(np.clip(norm_pos, -1, 1))
    
    return normalized


def denormalize_joint_positions(normalized_positions: List[float], limits: List[Tuple[float, float]]) -> List[float]:
    """
    Denormalize joint positions from [-1, 1] range to actual joint values.
    
    Args:
        normalized_positions: Positions in [-1, 1] range
        limits: Joint limits as [(min, max), ...]
        
    Returns:
        Actual joint positions
    """
    positions = []
    for norm_pos, (min_val, max_val) in zip(normalized_positions, limits):
        if min_val == -np.inf or max_val == np.inf:
            # For joints without limits, assume they're already in correct range
            positions.append(norm_pos)
        else:
            # Denormalize from [-1, 1]
            range_size = max_val - min_val
            center = (max_val + min_val) / 2
            pos = center + norm_pos * range_size / 2
            positions.append(np.clip(pos, min_val, max_val))
    
    return positions


def smooth_trajectory(waypoints: List[List[float]], num_interpolated: int = 10) -> List[List[float]]:
    """
    Generate smooth trajectory between waypoints using linear interpolation.
    
    Args:
        waypoints: List of waypoints, each containing joint positions
        num_interpolated: Number of interpolated points between each pair of waypoints
        
    Returns:
        Smooth trajectory as list of joint positions
    """
    if len(waypoints) < 2:
        return waypoints
    
    trajectory = []
    
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i + 1])
        
        # Add start waypoint
        trajectory.append(start.tolist())
        
        # Add interpolated points
        for j in range(1, num_interpolated + 1):
            alpha = j / (num_interpolated + 1)
            interpolated = start + alpha * (end - start)
            trajectory.append(interpolated.tolist())
    
    # Add final waypoint
    trajectory.append(waypoints[-1])
    
    return trajectory


def calculate_workspace_bounds(joint_limits: List[Tuple[float, float]], num_samples: int = 1000) -> Dict:
    """
    Calculate approximate workspace bounds for a robot arm.
    
    Args:
        joint_limits: Joint limits as [(min, max), ...]
        num_samples: Number of random samples to use for estimation
        
    Returns:
        Dictionary with workspace bounds
    """
    # This is a simplified estimation - in practice, you'd use forward kinematics
    # For now, we'll return reasonable default bounds
    
    workspace_bounds = {
        'x_min': -1.0,
        'x_max': 1.0,
        'y_min': -1.0,
        'y_max': 1.0,
        'z_min': 0.0,
        'z_max': 1.5
    }
    
    return workspace_bounds


def save_trajectory_data(trajectory: List[Dict], filename: str):
    """
    Save trajectory data to file.
    
    Args:
        trajectory: List of trajectory points, each containing robot state
        filename: Output filename
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_trajectory = []
    for point in trajectory:
        json_point = {}
        for key, value in point.items():
            if isinstance(value, np.ndarray):
                json_point[key] = value.tolist()
            else:
                json_point[key] = value
        json_trajectory.append(json_point)
    
    with open(filename, 'w') as f:
        json.dump(json_trajectory, f, indent=2)
    
    print(f"Trajectory saved to {filename}")


def load_trajectory_data(filename: str) -> List[Dict]:
    """
    Load trajectory data from file.
    
    Args:
        filename: Input filename
        
    Returns:
        List of trajectory points
    """
    import json
    
    with open(filename, 'r') as f:
        trajectory = json.load(f)
    
    print(f"Trajectory loaded from {filename}")
    return trajectory


def plot_joint_trajectory(joint_trajectories: List[List[float]], joint_names: List[str] = None):
    """
    Plot joint trajectories over time.
    
    Args:
        joint_trajectories: List of joint positions over time
        joint_names: Names of joints for labeling
    """
    if not joint_trajectories:
        return
    
    num_joints = len(joint_trajectories[0])
    time_steps = range(len(joint_trajectories))
    
    plt.figure(figsize=(12, 8))
    
    for i in range(num_joints):
        joint_values = [traj[i] for traj in joint_trajectories]
        label = f"Joint {i}" if joint_names is None else joint_names[i]
        plt.plot(time_steps, joint_values, label=label)
    
    plt.xlabel('Time Step')
    plt.ylabel('Joint Position (rad)')
    plt.title('Joint Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_workspace_coverage(positions: List[List[float]], title: str = "Workspace Coverage"):
    """
    Visualize 3D workspace coverage from end effector positions.
    
    Args:
        positions: List of [x, y, z] positions
        title: Plot title
    """
    if not positions:
        return
    
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               alpha=0.6, s=20, c=range(len(positions)), cmap='viridis')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    plt.show()


def apply_image_filters(image: np.ndarray, filter_type: str = "none") -> np.ndarray:
    """
    Apply image filters for vision processing.
    
    Args:
        image: Input image as numpy array
        filter_type: Type of filter to apply
        
    Returns:
        Filtered image
    """
    if filter_type == "none":
        return image
    elif filter_type == "blur":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "edge":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif filter_type == "hsv":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        print(f"Unknown filter type: {filter_type}")
        return image


def calculate_motion_smoothness(trajectory: List[List[float]]) -> float:
    """
    Calculate smoothness metric for a trajectory.
    
    Args:
        trajectory: List of joint positions over time
        
    Returns:
        Smoothness score (lower is smoother)
    """
    if len(trajectory) < 3:
        return 0.0
    
    trajectory = np.array(trajectory)
    
    # Calculate second derivatives (acceleration)
    accelerations = []
    for i in range(1, len(trajectory) - 1):
        accel = trajectory[i+1] - 2*trajectory[i] + trajectory[i-1]
        accelerations.append(np.linalg.norm(accel))
    
    # Return mean acceleration magnitude as smoothness metric
    return np.mean(accelerations)


def create_safety_bounds(center: List[float], size: List[float]) -> Dict:
    """
    Create safety bounds for robot operation.
    
    Args:
        center: Center point [x, y, z]
        size: Size of safe region [dx, dy, dz]
        
    Returns:
        Safety bounds dictionary
    """
    bounds = {
        'x_min': center[0] - size[0]/2,
        'x_max': center[0] + size[0]/2,
        'y_min': center[1] - size[1]/2,
        'y_max': center[1] + size[1]/2,
        'z_min': center[2] - size[2]/2,
        'z_max': center[2] + size[2]/2
    }
    
    return bounds


def check_safety_bounds(position: List[float], bounds: Dict) -> bool:
    """
    Check if position is within safety bounds.
    
    Args:
        position: [x, y, z] position to check
        bounds: Safety bounds dictionary
        
    Returns:
        True if position is safe
    """
    x, y, z = position
    
    return (bounds['x_min'] <= x <= bounds['x_max'] and
            bounds['y_min'] <= y <= bounds['y_max'] and
            bounds['z_min'] <= z <= bounds['z_max'])