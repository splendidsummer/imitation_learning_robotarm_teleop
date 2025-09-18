#!/usr/bin/env python3

"""
Setup script for imitation_learning_lerobot package.
This package provides tools for imitation learning using robotic manipulation environments.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from package if available
def get_version():
    version_file = this_directory / "imitation_learning_lerobot" / "__version__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            exec(f.read())
            return locals().get('__version__', '0.1.0')
    return '0.1.0'

# Core dependencies
install_requires = [
    # Core scientific computing
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    
    # Computer vision and image processing
    "opencv-python>=4.5.0",
    
    # Data handling
    "h5py>=3.1.0",
    
    # Robotics libraries
    "mujoco>=3.0.0",
    "roboticstoolbox-python>=1.0.0",
    "spatialmath-python>=1.0.0",
    "modern-robotics>=1.0.0",
    
    # Game controller support
    "pygame>=2.0.0",
    
    # Utility libraries
    "loop-rate-limiters>=1.0.0",
    
    # LeRobot integration (if available)
    # Note: This might need to be installed separately if not on PyPI
    # "lerobot",
]

# Optional dependencies for different features
extras_require = {
    'dev': [
        'pytest>=6.0',
        'pytest-cov',
        'black',
        'flake8',
        'isort',
        'mypy',
    ],
    'docs': [
        'sphinx>=4.0',
        'sphinx-rtd-theme',
        'sphinx-autodoc-typehints',
    ],
    'visualization': [
        'matplotlib>=3.3.0',
        'plotly>=5.0.0',
    ],
    'joycon': [
        'joycon-python>=0.2.0',  # For Nintendo Joy-Con support
    ],
}

# All optional dependencies
extras_require['all'] = sum(extras_require.values(), [])

setup(
    name="imitation_learning_lerobot",
    version=get_version(),
    description="Imitation Learning toolkit integrated with LeRobot for robotic manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LeRobot Team",
    author_email="lerobot@example.com",  # Replace with actual email
    url="https://github.com/yourusername/imitation_learning_lerobot",  # Replace with actual URL
    
    # Package discovery
    packages=find_packages(include=["imitation_learning_lerobot", "imitation_learning_lerobot.*"]),
    
    # Include package data
    include_package_data=True,
    package_data={
        'imitation_learning_lerobot': [
            'assets/**/*',
            'configs/*.py',
            'scripts/*.py',
        ],
    },
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'collect_data_teleoperation=imitation_learning_lerobot.scripts.collect_data_teleoperation:main',
            'collect_data=imitation_learning_lerobot.scripts.collect_data:main',
            'convert_h5_to_lerobot=imitation_learning_lerobot.scripts.convert_h5_to_lerobot:main',
            'rollout=imitation_learning_lerobot.scripts.rollout:main',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Change if different license
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for discoverability
    keywords=[
        "robotics",
        "imitation learning", 
        "machine learning",
        "manipulation",
        "mujoco",
        "lerobot",
        "teleoperation",
        "data collection",
    ],
    
    # License
    license="MIT",  # Change if different license
    
    # Additional metadata
    project_urls={
        "Bug Reports": "https://github.com/yourusername/imitation_learning_lerobot/issues",
        "Source": "https://github.com/yourusername/imitation_learning_lerobot",
        "Documentation": "https://github.com/yourusername/imitation_learning_lerobot/blob/main/README.md",
    },
    
    # Zip safe
    zip_safe=False,
)
