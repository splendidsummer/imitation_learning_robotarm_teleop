# Robot Arm Teleoperation for Imitation Learning

A comprehensive Python package for robot arm simulation, teleoperation, and imitation learning. This package provides simulation environments and teleoperation control for robot arms using joysticks, keyboards, and other input devices, with specific implementations for ALOHA-style bimanual manipulation.

## Features

- **Robot Arm Simulation**: Physics-based simulation using PyBullet
- **Multiple Input Devices**: Support for keyboard, joystick/gamepad control
- **Control Modes**: Joint-space and Cartesian-space teleoperation
- **ALOHA Environment**: Bimanual robot setup for imitation learning
- **Visualization**: Real-time 3D visualization and camera feeds
- **Extensible Architecture**: Easy to add new robots and tasks

## Installation

### Prerequisites

- Python 3.8 or higher
- PyBullet for physics simulation
- Pygame for input handling
- NumPy, OpenCV, and other scientific computing libraries

### Install from Source

```bash
git clone https://github.com/splendidsummer/imitation_learning_robotarm_teleop.git
cd imitation_learning_robotarm_teleop
pip install -r requirements.txt
pip install -e .
```

### Dependencies

The package requires the following main dependencies:

- `pybullet>=3.2.5` - Physics simulation
- `pygame>=2.0.0` - Input device handling
- `numpy>=1.20.0` - Numerical computations
- `opencv-python>=4.5.0` - Computer vision
- `matplotlib>=3.3.0` - Plotting and visualization

## Quick Start

### Basic Robot Arm Simulation

```python
from robot_arm_teleop.simulation import RobotArmSimulation

# Create robot arm simulation
robot_sim = RobotArmSimulation(use_gui=True)

# Move robot to a target position
target_positions = [0.5, -0.3, 0.8, 0.0, 0.0, 0.0]
robot_sim.move_to_joint_positions(target_positions)

# Step simulation
for _ in range(1000):
    robot_sim.step_simulation()
```

### Keyboard Teleoperation

```python
from robot_arm_teleop.simulation import RobotArmSimulation
from robot_arm_teleop.teleoperation import TeleoperationController, KeyboardInput

# Setup
robot_sim = RobotArmSimulation(use_gui=True)
keyboard = KeyboardInput()
teleop = TeleoperationController(robot_sim, keyboard, control_mode="joint")

# Run teleoperation
teleop.run_main_loop()
```

### ALOHA Bimanual Environment

```python
from robot_arm_teleop.environments import ALOHAEnvironment

# Create ALOHA environment
aloha_env = ALOHAEnvironment(use_gui=True)

# Add task objects
aloha_env.add_task_objects("pick_and_place")

# Control both arms
left_positions = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
right_positions = [-0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

aloha_env.control_left_arm(left_positions)
aloha_env.control_right_arm(right_positions)
```

## Package Structure

```
robot_arm_teleop/
├── simulation/           # Core simulation components
│   ├── robot_arm_simulation.py
│   └── __init__.py
├── teleoperation/       # Teleoperation controllers
│   ├── teleoperation_controller.py
│   └── __init__.py
├── environments/        # Specific robot environments
│   ├── aloha_environment.py
│   └── __init__.py
├── utils/              # Utility functions
│   ├── math_utils.py
│   └── __init__.py
└── __init__.py

examples/               # Example scripts
├── basic_simulation.py
├── keyboard_teleop.py
└── aloha_demo.py

tests/                 # Unit tests
└── test_robot_arm_teleop.py
```

## Examples

The `examples/` directory contains several demonstration scripts:

### 1. Basic Simulation (`examples/basic_simulation.py`)

Demonstrates basic robot arm simulation features:
- Robot loading and joint control
- End effector positioning
- Inverse kinematics
- Camera image capture

```bash
python examples/basic_simulation.py
```

### 2. Keyboard Teleoperation (`examples/keyboard_teleop.py`)

Interactive keyboard-based robot control:

**Controls:**
- `Q/W`: Joint 0, `A/S`: Joint 1, `Z/X`: Joint 2
- `E/R`: Joint 3, `D/F`: Joint 4, `C/V`: Joint 5
- Arrow keys: End effector X/Y, Page Up/Down: Z
- `SPACE`: Pause/Resume, `H`: Home position, `ESC`: Exit

```bash
python examples/keyboard_teleop.py
```

### 3. ALOHA Demo (`examples/aloha_demo.py`)

Demonstrates bimanual robot manipulation:
- Pick and place tasks
- Bimanual coordination
- Object stacking
- Multi-camera views

```bash
python examples/aloha_demo.py
```

## API Reference

### RobotArmSimulation

Main class for robot arm physics simulation.

#### Key Methods:

- `__init__(robot_urdf=None, use_gui=True)`: Initialize simulation
- `move_to_joint_positions(positions)`: Move robot to joint positions
- `get_joint_states()`: Get current joint positions and velocities
- `get_end_effector_pose()`: Get end effector position and orientation
- `inverse_kinematics(target_position, target_orientation=None)`: Compute IK solution
- `get_camera_image(width=640, height=480)`: Capture camera image
- `step_simulation()`: Advance physics simulation by one step

### TeleoperationController

Handles teleoperation using various input devices.

#### Key Methods:

- `__init__(robot_simulation, input_device, control_mode="joint")`: Initialize controller
- `set_control_mode(mode)`: Switch between "joint" and "cartesian" control
- `start()`: Start teleoperation
- `stop()`: Stop teleoperation
- `run_main_loop()`: Run main teleoperation loop

### ALOHAEnvironment

Bimanual robot environment for ALOHA-style manipulation.

#### Key Methods:

- `__init__(use_gui=True)`: Initialize ALOHA environment
- `control_left_arm(joint_positions)`: Control left robot arm
- `control_right_arm(joint_positions)`: Control right robot arm
- `add_task_objects(task_name)`: Add objects for specific tasks
- `get_robot_states()`: Get states of both robot arms
- `get_camera_images()`: Capture multi-view camera images

## Control Modes

### Joint Control Mode

Direct control of individual robot joints:
- Each joint can be controlled independently
- Good for precise manipulation and robot familiarization
- Maps directly to robot joint space

### Cartesian Control Mode

Control robot end effector in Cartesian space:
- Move end effector in X, Y, Z directions
- More intuitive for manipulation tasks
- Uses inverse kinematics internally

## Supported Input Devices

### Keyboard

- **Pros**: Always available, precise control
- **Cons**: Limited simultaneous inputs
- **Best for**: Development, testing, precise movements

### Joystick/Gamepad

- **Pros**: Analog control, many buttons, natural feel
- **Cons**: Requires hardware, driver setup
- **Best for**: Smooth teleoperation, real-time control

## Tasks and Environments

### Pick and Place

Simple manipulation tasks with individual objects:
- Single-arm grasping and placement
- Object detection and tracking
- Basic manipulation primitives

### Bimanual Manipulation

Coordinated two-arm manipulation:
- Large object handling
- Assembly tasks
- Cooperative manipulation

### Stacking

Precision manipulation for object stacking:
- Fine motor control
- Spatial reasoning
- Sequential manipulation

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

Or run specific test classes:

```bash
python tests/test_robot_arm_teleop.py
```

### Adding New Robots

1. Create or obtain robot URDF file
2. Initialize RobotArmSimulation with custom URDF:

```python
robot_sim = RobotArmSimulation(robot_urdf="path/to/robot.urdf")
```

### Adding New Tasks

1. Extend ALOHAEnvironment or create new environment class
2. Implement task-specific object placement and goals
3. Add to task selection in `add_task_objects()` method

### Adding New Input Devices

1. Create new class inheriting from `InputDevice`
2. Implement `get_input()` and `is_connected()` methods
3. Add device-specific processing in TeleoperationController

## Troubleshooting

### Common Issues

**PyBullet Installation Issues:**
```bash
pip install --upgrade pybullet
```

**Pygame Audio Issues:**
```bash
export SDL_AUDIODRIVER=dummy  # Linux/Mac
set SDL_AUDIODRIVER=dummy     # Windows
```

**Robot Not Loading:**
- Check URDF file path and format
- Ensure all mesh files are available
- Try with default robot first

**Input Device Not Working:**
- For joystick: Check device connection and drivers
- For keyboard: Ensure PyGame display is focused
- Check device permissions on Linux

### Performance Tips

- Use `use_gui=False` for faster simulation in headless mode
- Reduce physics time step for faster-than-real-time simulation
- Lower camera resolution for better performance
- Disable unnecessary visual elements

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyBullet team for the excellent physics simulation
- ALOHA project for bimanual manipulation inspiration
- Open source robotics community

## Citation

If you use this package in your research, please cite:

```bibtex
@software{robot_arm_teleop,
  title={Robot Arm Teleoperation for Imitation Learning},
  author={splendidsummer},
  url={https://github.com/splendidsummer/imitation_learning_robotarm_teleop},
  year={2025}
}
``` 
