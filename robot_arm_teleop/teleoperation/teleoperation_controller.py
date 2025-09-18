"""
Teleoperation controller for robot arms supporting various input devices.
"""

import pygame
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod


class InputDevice(ABC):
    """Abstract base class for input devices."""
    
    @abstractmethod
    def get_input(self) -> Dict:
        """Get current input state from device."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected."""
        pass


class KeyboardInput(InputDevice):
    """Keyboard input device for teleoperation."""
    
    def __init__(self):
        pygame.init()
        self.keys_pressed = set()
        
        # Key mappings for robot control
        self.key_mappings = {
            # Joint control
            pygame.K_q: "joint_0_neg",
            pygame.K_w: "joint_0_pos", 
            pygame.K_a: "joint_1_neg",
            pygame.K_s: "joint_1_pos",
            pygame.K_z: "joint_2_neg",
            pygame.K_x: "joint_2_pos",
            pygame.K_e: "joint_3_neg",
            pygame.K_r: "joint_3_pos",
            pygame.K_d: "joint_4_neg",
            pygame.K_f: "joint_4_pos",
            pygame.K_c: "joint_5_neg",
            pygame.K_v: "joint_5_pos",
            
            # End effector control
            pygame.K_UP: "ee_x_pos",
            pygame.K_DOWN: "ee_x_neg",
            pygame.K_LEFT: "ee_y_neg", 
            pygame.K_RIGHT: "ee_y_pos",
            pygame.K_PAGEUP: "ee_z_pos",
            pygame.K_PAGEDOWN: "ee_z_neg",
            
            # Special commands
            pygame.K_SPACE: "stop",
            pygame.K_h: "home",
            pygame.K_ESCAPE: "quit"
        }
    
    def get_input(self) -> Dict:
        """Get current keyboard input state."""
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
        
        # Convert pressed keys to control commands
        commands = {}
        for key in self.keys_pressed:
            if key in self.key_mappings:
                commands[self.key_mappings[key]] = True
        
        return commands
    
    def is_connected(self) -> bool:
        """Keyboard is always available."""
        return True


class JoystickInput(InputDevice):
    """Joystick/gamepad input device for teleoperation."""
    
    def __init__(self, joystick_id: int = 0):
        pygame.init()
        pygame.joystick.init()
        
        self.joystick_id = joystick_id
        self.joystick = None
        
        # Initialize joystick if available
        if pygame.joystick.get_count() > joystick_id:
            self.joystick = pygame.joystick.Joystick(joystick_id)
            self.joystick.init()
            print(f"Initialized joystick: {self.joystick.get_name()}")
        else:
            print(f"No joystick found at index {joystick_id}")
    
    def get_input(self) -> Dict:
        """Get current joystick input state."""
        if not self.is_connected():
            return {}
        
        # Process pygame events
        pygame.event.pump()
        
        commands = {}
        
        # Get axis values (typically -1 to 1)
        num_axes = self.joystick.get_numaxes()
        for i in range(min(num_axes, 6)):  # Limit to 6 axes
            axis_value = self.joystick.get_axis(i)
            # Apply deadzone
            if abs(axis_value) > 0.1:
                commands[f"axis_{i}"] = axis_value
        
        # Get button states
        num_buttons = self.joystick.get_numbuttons()
        for i in range(num_buttons):
            if self.joystick.get_button(i):
                commands[f"button_{i}"] = True
        
        # Get hat (D-pad) states
        num_hats = self.joystick.get_numhats()
        for i in range(num_hats):
            hat_x, hat_y = self.joystick.get_hat(i)
            if hat_x != 0:
                commands[f"hat_{i}_x"] = hat_x
            if hat_y != 0:
                commands[f"hat_{i}_y"] = hat_y
        
        return commands
    
    def is_connected(self) -> bool:
        """Check if joystick is connected."""
        return self.joystick is not None and self.joystick.get_init()


class TeleoperationController:
    """
    Main teleoperation controller that manages input devices and robot control.
    """
    
    def __init__(
        self,
        robot_simulation,
        input_device: InputDevice = None,
        control_mode: str = "joint",
        update_rate: float = 60.0
    ):
        """
        Initialize teleoperation controller.
        
        Args:
            robot_simulation: RobotArmSimulation instance
            input_device: Input device to use (keyboard by default)
            control_mode: "joint" or "cartesian" control mode
            update_rate: Control update rate in Hz
        """
        self.robot_sim = robot_simulation
        self.control_mode = control_mode
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        
        # Set up input device
        if input_device is None:
            self.input_device = KeyboardInput()
        else:
            self.input_device = input_device
        
        # Control parameters
        self.joint_velocity_scale = 1.0  # rad/s per input unit
        self.cartesian_velocity_scale = 0.1  # m/s per input unit
        self.angular_velocity_scale = 1.0  # rad/s per input unit
        
        # Current target state
        self.target_joint_positions = []
        self.target_ee_position = [0, 0, 0]
        self.target_ee_orientation = [0, 0, 0, 1]
        
        # Control state
        self.is_running = False
        self.paused = False
        
        # Initialize target positions
        self._initialize_targets()
        
        print("Teleoperation controller initialized!")
        self._print_help()
    
    def _initialize_targets(self):
        """Initialize target positions from current robot state."""
        positions, _ = self.robot_sim.get_joint_states()
        self.target_joint_positions = positions.copy()
        
        ee_pos, ee_ori = self.robot_sim.get_end_effector_pose()
        self.target_ee_position = ee_pos.copy()
        self.target_ee_orientation = ee_ori.copy()
    
    def _print_help(self):
        """Print control help information."""
        print("\\n=== Teleoperation Controls ===")
        if isinstance(self.input_device, KeyboardInput):
            print("Keyboard Controls:")
            print("  Joints: Q/W (J0), A/S (J1), Z/X (J2), E/R (J3), D/F (J4), C/V (J5)")
            print("  End Effector: Arrow keys (X/Y), Page Up/Down (Z)")
            print("  Commands: SPACE (stop), H (home), ESC (quit)")
        elif isinstance(self.input_device, JoystickInput):
            print("Joystick Controls:")
            print("  Left stick: X/Y movement")
            print("  Right stick: Z movement and rotation")
            print("  Buttons: Various functions")
        print("  Control Mode:", self.control_mode)
        print("===============================\\n")
    
    def set_control_mode(self, mode: str):
        """Set control mode ('joint' or 'cartesian')."""
        if mode in ["joint", "cartesian"]:
            self.control_mode = mode
            print(f"Control mode set to: {mode}")
        else:
            print(f"Invalid control mode: {mode}")
    
    def process_joint_control(self, commands: Dict):
        """Process commands for joint-space control."""
        joint_velocities = [0.0] * len(self.target_joint_positions)
        
        if isinstance(self.input_device, KeyboardInput):
            # Keyboard joint control
            for i in range(len(joint_velocities)):
                pos_key = f"joint_{i}_pos"
                neg_key = f"joint_{i}_neg"
                
                if commands.get(pos_key, False):
                    joint_velocities[i] = self.joint_velocity_scale
                elif commands.get(neg_key, False):
                    joint_velocities[i] = -self.joint_velocity_scale
        
        elif isinstance(self.input_device, JoystickInput):
            # Joystick joint control
            # Map axes to joints
            for i in range(min(len(joint_velocities), 6)):
                axis_key = f"axis_{i}"
                if axis_key in commands:
                    joint_velocities[i] = commands[axis_key] * self.joint_velocity_scale
        
        # Update target joint positions
        for i in range(len(self.target_joint_positions)):
            self.target_joint_positions[i] += joint_velocities[i] * self.dt
            
            # Clamp to joint limits
            if i < len(self.robot_sim.joint_limits):
                lower, upper = self.robot_sim.joint_limits[i]
                if lower != -np.inf and upper != np.inf:
                    self.target_joint_positions[i] = np.clip(
                        self.target_joint_positions[i], lower, upper
                    )
        
        # Send command to robot
        self.robot_sim.move_to_joint_positions(self.target_joint_positions)
    
    def process_cartesian_control(self, commands: Dict):
        """Process commands for Cartesian-space control."""
        ee_velocity = [0.0, 0.0, 0.0]
        angular_velocity = [0.0, 0.0, 0.0]
        
        if isinstance(self.input_device, KeyboardInput):
            # Keyboard Cartesian control
            if commands.get("ee_x_pos", False):
                ee_velocity[0] = self.cartesian_velocity_scale
            elif commands.get("ee_x_neg", False):
                ee_velocity[0] = -self.cartesian_velocity_scale
            
            if commands.get("ee_y_pos", False):
                ee_velocity[1] = self.cartesian_velocity_scale
            elif commands.get("ee_y_neg", False):
                ee_velocity[1] = -self.cartesian_velocity_scale
            
            if commands.get("ee_z_pos", False):
                ee_velocity[2] = self.cartesian_velocity_scale
            elif commands.get("ee_z_neg", False):
                ee_velocity[2] = -self.cartesian_velocity_scale
        
        elif isinstance(self.input_device, JoystickInput):
            # Joystick Cartesian control
            # Left stick for X/Y, right stick Y for Z
            if "axis_0" in commands:
                ee_velocity[1] = commands["axis_0"] * self.cartesian_velocity_scale
            if "axis_1" in commands:
                ee_velocity[0] = -commands["axis_1"] * self.cartesian_velocity_scale
            if "axis_3" in commands:
                ee_velocity[2] = -commands["axis_3"] * self.cartesian_velocity_scale
            
            # Right stick X for rotation around Z
            if "axis_2" in commands:
                angular_velocity[2] = commands["axis_2"] * self.angular_velocity_scale
        
        # Update target end effector position
        for i in range(3):
            self.target_ee_position[i] += ee_velocity[i] * self.dt
        
        # Compute inverse kinematics
        try:
            target_joints = self.robot_sim.inverse_kinematics(
                self.target_ee_position, self.target_ee_orientation
            )
            self.target_joint_positions = target_joints
            self.robot_sim.move_to_joint_positions(target_joints)
        except Exception as e:
            print(f"IK failed: {e}")
    
    def process_special_commands(self, commands: Dict):
        """Process special commands (stop, home, etc.)."""
        if commands.get("stop", False):
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")
        
        if commands.get("home", False):
            print("Moving to home position...")
            self.robot_sim.reset_robot_pose()
            self._initialize_targets()
        
        if commands.get("quit", False):
            print("Stopping teleoperation...")
            self.stop()
    
    def update(self):
        """Update teleoperation control (call this in main loop)."""
        if not self.is_running or self.paused:
            return
        
        # Get input from device
        commands = self.input_device.get_input()
        
        if not commands:
            return
        
        # Process special commands first
        self.process_special_commands(commands)
        
        if self.paused:
            return
        
        # Process movement commands based on control mode
        if self.control_mode == "joint":
            self.process_joint_control(commands)
        elif self.control_mode == "cartesian":
            self.process_cartesian_control(commands)
    
    def start(self):
        """Start teleoperation control."""
        self.is_running = True
        print("Teleoperation started!")
    
    def stop(self):
        """Stop teleoperation control."""
        self.is_running = False
        print("Teleoperation stopped!")
    
    def run_main_loop(self):
        """Run the main teleoperation loop."""
        self.start()
        
        try:
            clock = pygame.time.Clock()
            
            while self.is_running:
                # Update control
                self.update()
                
                # Step simulation
                self.robot_sim.step_simulation()
                
                # Maintain update rate
                clock.tick(self.update_rate)
                
        except KeyboardInterrupt:
            print("\\nKeyboard interrupt received!")
        finally:
            self.stop()


# Export classes
from .teleoperation_controller import TeleoperationController, KeyboardInput, JoystickInput

__all__ = ["TeleoperationController", "KeyboardInput", "JoystickInput"]