# Installation Guide

## Quick Installation

### From Source (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imitation_learning_lerobot.git
cd imitation_learning_lerobot
```

2. Install in development mode:
```bash
pip install -e .
```

3. Or install with all optional dependencies:
```bash
pip install -e ".[all]"
```

### From PyPI (when available)

```bash
pip install imitation_learning_lerobot
```

## Installation Options

### Core Installation
```bash
pip install imitation_learning_lerobot
```

### Development Installation
```bash
pip install imitation_learning_lerobot[dev]
```

### With Joy-Con Support
```bash
pip install imitation_learning_lerobot[joycon]
```

### With Visualization Tools
```bash
pip install imitation_learning_lerobot[visualization]
```

### Complete Installation
```bash
pip install imitation_learning_lerobot[all]
```

## Dependencies

### Core Dependencies
- numpy >= 1.21.0
- scipy >= 1.7.0
- opencv-python >= 4.5.0
- h5py >= 3.1.0
- mujoco >= 3.0.0
- roboticstoolbox-python >= 1.0.0
- spatialmath-python >= 1.0.0
- modern-robotics >= 1.0.0
- pygame >= 2.0.0
- loop-rate-limiters >= 1.0.0

### Optional Dependencies
- **Development**: pytest, black, flake8, mypy
- **Documentation**: sphinx, sphinx-rtd-theme
- **Visualization**: matplotlib, plotly
- **Joy-Con Support**: joycon-python

## System Requirements

- Python 3.8 or higher
- Operating System: Linux, macOS, Windows
- For MuJoCo: Compatible graphics drivers

## Troubleshooting

### MuJoCo Installation Issues
If you encounter MuJoCo installation issues:
```bash
# On Ubuntu/Debian
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3

# On macOS
brew install glfw3
```

### Joy-Con Connection Issues
For Nintendo Joy-Con support, ensure you have proper Bluetooth permissions and drivers.

### Graphics Issues
For headless environments, set:
```bash
export MUJOCO_GL=osmesa
```

## Verification

Test your installation:
```python
import imitation_learning_lerobot
print(imitation_learning_lerobot.__version__)
```

## Uninstallation

```bash
pip uninstall imitation_learning_lerobot
```
