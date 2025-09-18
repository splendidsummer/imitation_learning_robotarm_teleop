# Robot Arm Simulation for Imitation Learning and Teleoperation

A simple robot arm simulation environment for imitation learning and teleoperation control by joystick and other controllers, implementing ALOHA-style robots and more.

## Features

- **3-DOF Robot Arm Simulation**: Simple 3-degree-of-freedom robot arm with configurable link lengths
- **Forward & Inverse Kinematics**: Complete kinematic calculations for precise positioning
- **Real-time Visualization**: Interactive matplotlib-based visualization with trajectory tracking
- **Control Interface**: Support for custom control callbacks and target positioning
- **Workspace Analysis**: Automatic workspace boundary calculation and reachability testing
- **Interactive Demo**: Mouse-controlled interface for manual target setting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/splendidsummer/imitation_learning_robotarm_teleop.git
cd imitation_learning_robotarm_teleop
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from robot_arm_sim import RobotArm, RobotArmSimulation
import numpy as np

# Create a robot arm
robot = RobotArm(link_lengths=[1.0, 0.8, 0.6])

# Set joint angles (in radians)
robot.set_joint_angles([np.pi/4, np.pi/6, -np.pi/8])

# Get end effector position
end_pos, joint_positions = robot.forward_kinematics()
print(f"End effector at: ({end_pos[0]:.2f}, {end_pos[1]:.2f})")

# Move to target position using inverse kinematics
success = robot.move_to_position(1.2, 0.8)
if success:
    print("Target reached!")
```

### Interactive Simulation

```python
# Create simulation with visualization
robot = RobotArm()
sim = RobotArmSimulation(robot)

# Run interactive mode (requires display)
# Left click to set targets, 'R' to reset, 'C' to clear trajectory
sim.run_interactive()
```

### Custom Control

```python
def circle_control(simulation):
    """Move end effector in a circle."""
    t = simulation.time
    radius = 1.0
    center_x, center_y = 0.5, 0.5
    
    target_x = center_x + radius * np.cos(t)
    target_y = center_y + radius * np.sin(t)
    
    simulation.set_target_position(target_x, target_y)
    simulation.move_to_target()

sim = RobotArmSimulation()
sim.set_control_callback(circle_control)
sim.run_animation(duration=8.0)
```

## Running Demos

The repository includes comprehensive demos:

```bash
python demo.py
```

Available demos:
1. **Basic Kinematics Demo** - Test forward/inverse kinematics
2. **Interactive Simulation** - Mouse-controlled robot arm
3. **Predefined Movement Demo** - Automated movement patterns
4. **Custom Control Demo** - Circular trajectory example
5. **Workspace Analysis** - Reachability testing
6. **Run All Demos** - Execute all demonstrations

## API Reference

### RobotArm Class

#### Constructor
- `RobotArm(link_lengths=[1.0, 0.8, 0.6])` - Create robot with specified link lengths

#### Methods
- `set_joint_angles(angles)` - Set joint angles with limit checking
- `get_joint_angles()` - Get current joint angles
- `forward_kinematics()` - Compute end effector and joint positions
- `inverse_kinematics_2d(x, y)` - 2-DOF inverse kinematics
- `move_to_position(x, y)` - Move to target using IK
- `get_workspace_bounds()` - Get workspace boundaries

### RobotArmSimulation Class

#### Constructor
- `RobotArmSimulation(robot_arm=None, update_rate=50.0)` - Create simulation environment

#### Methods
- `setup_visualization(figsize=(10, 8))` - Initialize matplotlib visualization
- `run_interactive()` - Run interactive mouse-controlled simulation
- `run_animation(duration=10.0)` - Run animated simulation
- `set_control_callback(callback)` - Set custom control function
- `set_target_position(x, y)` - Set target position
- `step()` - Perform one simulation step
- `reset()` - Reset to initial state

## Testing

Run the test suite to validate functionality:

```bash
python test_robot_arm.py
```

## Robot Specifications

- **Degrees of Freedom**: 3 (all revolute joints)
- **Default Link Lengths**: [1.0, 0.8, 0.6] units
- **Joint Limits**: 
  - Joint 1: ±180° (full rotation)
  - Joint 2: ±90° (limited range)
  - Joint 3: ±90° (limited range)
- **Workspace**: Circular with radius equal to sum of link lengths

## Requirements

- Python 3.7+
- NumPy ≥ 1.20.0
- Matplotlib ≥ 3.5.0
- SciPy ≥ 1.7.0

## Future Extensions

This simulation environment is designed to be extended for:

- **ALOHA-style dual arm robots**
- **Joystick/gamepad teleoperation**
- **Imitation learning data collection**
- **Motion planning algorithms**
- **Physics simulation integration**
- **ROS integration**

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements. 
