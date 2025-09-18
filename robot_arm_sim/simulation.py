"""
Robot Arm Simulation Environment with visualization and control interface.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Callable
import time

from .robot_arm import RobotArm


class RobotArmSimulation:
    """
    Robot arm simulation environment with real-time visualization and control.
    """
    
    def __init__(self, robot_arm: RobotArm = None, update_rate: float = 50.0):
        """
        Initialize the simulation environment.
        
        Args:
            robot_arm: RobotArm instance. If None, creates default arm.
            update_rate: Simulation update rate in Hz
        """
        self.robot_arm = robot_arm if robot_arm is not None else RobotArm()
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        
        # Simulation state
        self.is_running = False
        self.time = 0.0
        
        # Visualization setup
        self.fig = None
        self.ax = None
        self.animation = None
        
        # Plotting elements
        self.arm_lines = []
        self.joint_circles = []
        self.end_effector_circle = None
        self.target_marker = None
        
        # Target position for demonstrations
        self.target_position = None
        
        # Trajectory recording
        self.trajectory = []
        self.trajectory_line = None
        
        # Control callback
        self.control_callback = None
        
    def setup_visualization(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Setup matplotlib visualization.
        
        Args:
            figsize: Figure size (width, height)
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Get workspace bounds
        min_x, max_x, min_y, max_y = self.robot_arm.get_workspace_bounds()
        margin = 0.2
        self.ax.set_xlim(min_x - margin, max_x + margin)
        self.ax.set_ylim(min_y - margin, max_y + margin)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Robot Arm Simulation')
        
        # Initialize visualization elements
        self._init_plot_elements()
        
    def _init_plot_elements(self) -> None:
        """Initialize plotting elements."""
        # Arm links
        for i in range(self.robot_arm.num_joints):
            line, = self.ax.plot([], [], 'b-', linewidth=4, alpha=0.8)
            self.arm_lines.append(line)
        
        # Joints
        for i in range(self.robot_arm.num_joints + 1):  # +1 for base
            circle = patches.Circle((0, 0), 0.05, color='red', zorder=10)
            self.ax.add_patch(circle)
            self.joint_circles.append(circle)
        
        # End effector
        self.end_effector_circle = patches.Circle((0, 0), 0.08, color='green', zorder=15)
        self.ax.add_patch(self.end_effector_circle)
        
        # Target position marker
        self.target_marker, = self.ax.plot([], [], 'rx', markersize=15, markeredgewidth=3)
        
        # Trajectory line
        self.trajectory_line, = self.ax.plot([], [], 'g--', alpha=0.5, linewidth=1)
        
    def update_visualization(self) -> None:
        """Update the visualization with current robot state."""
        if self.ax is None:
            return
        
        # Get current robot configuration
        end_effector_pos, joint_positions = self.robot_arm.forward_kinematics()
        links = self.robot_arm.get_link_positions()
        
        # Update arm links
        for i, (start_pos, end_pos) in enumerate(links):
            self.arm_lines[i].set_data([start_pos[0], end_pos[0]], 
                                     [start_pos[1], end_pos[1]])
        
        # Update joint positions
        for i, pos in enumerate(joint_positions):
            self.joint_circles[i].center = (pos[0], pos[1])
        
        # Update end effector
        self.end_effector_circle.center = (end_effector_pos[0], end_effector_pos[1])
        
        # Update target marker
        if self.target_position is not None:
            self.target_marker.set_data([self.target_position[0]], [self.target_position[1]])
        
        # Update trajectory
        if len(self.trajectory) > 1:
            traj_x = [pos[0] for pos in self.trajectory]
            traj_y = [pos[1] for pos in self.trajectory]
            self.trajectory_line.set_data(traj_x, traj_y)
        
    def set_target_position(self, x: float, y: float) -> None:
        """
        Set target position for the robot arm.
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
        """
        self.target_position = np.array([x, y])
        
    def move_to_target(self, speed: float = 1.0) -> bool:
        """
        Move robot arm to target position.
        
        Args:
            speed: Movement speed factor (0.1 to 2.0)
            
        Returns:
            True if target reached, False if unreachable
        """
        if self.target_position is None:
            return False
        
        return self.robot_arm.move_to_position(self.target_position[0], self.target_position[1])
    
    def set_control_callback(self, callback: Callable[['RobotArmSimulation'], None]) -> None:
        """
        Set a control callback function that will be called each simulation step.
        
        Args:
            callback: Function that takes the simulation instance as parameter
        """
        self.control_callback = callback
    
    def step(self) -> None:
        """Perform one simulation step."""
        # Call control callback if set
        if self.control_callback is not None:
            self.control_callback(self)
        
        # Record trajectory
        end_effector_pos, _ = self.robot_arm.forward_kinematics()
        self.trajectory.append(end_effector_pos.copy())
        
        # Limit trajectory length
        if len(self.trajectory) > 1000:
            self.trajectory.pop(0)
        
        # Update time
        self.time += self.dt
        
        # Update visualization
        self.update_visualization()
    
    def run_interactive(self, duration: Optional[float] = None) -> None:
        """
        Run interactive simulation with mouse control.
        
        Args:
            duration: Simulation duration in seconds. If None, runs indefinitely.
        """
        if self.fig is None:
            self.setup_visualization()
        
        def on_click(event):
            if event.inaxes == self.ax and event.button == 1:  # Left click
                self.set_target_position(event.xdata, event.ydata)
                self.move_to_target()
        
        def on_key(event):
            if event.key == 'r':  # Reset
                self.reset()
            elif event.key == 'c':  # Clear trajectory
                self.trajectory.clear()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', on_click)
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Initial update
        self.update_visualization()
        
        # Add instructions
        self.fig.suptitle('Robot Arm Simulation\\nLeft click to set target, R to reset, C to clear trajectory')
        
        plt.show()
    
    def run_animation(self, duration: float = 10.0, save_path: Optional[str] = None) -> None:
        """
        Run simulation with animation.
        
        Args:
            duration: Animation duration in seconds
            save_path: If provided, save animation to this path
        """
        if self.fig is None:
            self.setup_visualization()
        
        self.is_running = True
        frames = int(duration * self.update_rate)
        
        def animate(frame):
            if self.is_running:
                self.step()
            return self.arm_lines + self.joint_circles + [self.end_effector_circle, 
                                                         self.target_marker, self.trajectory_line]
        
        self.animation = FuncAnimation(self.fig, animate, frames=frames, 
                                     interval=1000/self.update_rate, blit=False, repeat=False)
        
        if save_path:
            self.animation.save(save_path, writer='pillow', fps=self.update_rate)
        else:
            plt.show()
    
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.robot_arm.set_joint_angles([0, 0, 0])
        self.trajectory.clear()
        self.target_position = None
        self.time = 0.0
        
        if self.ax is not None:
            self.update_visualization()
    
    def demonstrate_movement(self) -> None:
        """Demonstrate robot arm movement with predefined targets."""
        if self.fig is None:
            self.setup_visualization()
        
        # Define demonstration points
        demo_points = [
            (1.0, 0.5),
            (0.8, 1.2),
            (-0.5, 1.0),
            (-1.2, 0.3),
            (0.0, 1.5),
            (1.5, -0.5)
        ]
        
        def demo_control(sim):
            # Change target every 2 seconds
            target_index = int(sim.time / 2.0) % len(demo_points)
            target = demo_points[target_index]
            sim.set_target_position(target[0], target[1])
            sim.move_to_target()
        
        self.set_control_callback(demo_control)
        self.run_animation(duration=len(demo_points) * 2.0)
    
    def get_state(self) -> dict:
        """
        Get current simulation state.
        
        Returns:
            Dictionary containing current state information
        """
        end_effector_pos, joint_positions = self.robot_arm.forward_kinematics()
        
        return {
            'time': self.time,
            'joint_angles': self.robot_arm.get_joint_angles(),
            'end_effector_position': end_effector_pos,
            'joint_positions': joint_positions,
            'target_position': self.target_position,
            'trajectory_length': len(self.trajectory)
        }
    
    def close(self) -> None:
        """Close the simulation and cleanup resources."""
        self.is_running = False
        if self.fig is not None:
            plt.close(self.fig)