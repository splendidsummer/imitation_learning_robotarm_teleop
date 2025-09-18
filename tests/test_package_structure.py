#!/usr/bin/env python3
"""
Test package installation and imports.

This script tests the basic package structure without requiring
all dependencies to be installed.
"""

import sys
import os

# Add the package to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic package imports...")
    
    try:
        import robot_arm_teleop
        print("✓ Package imported successfully")
        print(f"✓ Version: {robot_arm_teleop.__version__}")
        print(f"✓ Author: {robot_arm_teleop.__author__}")
        print(f"✓ Available classes: {robot_arm_teleop.__all__}")
        return True
    except Exception as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_class_access():
    """Test accessing classes through lazy imports."""
    print("\\nTesting class access...")
    
    try:
        import robot_arm_teleop
        
        # Test accessing each class
        classes_to_test = ["RobotArmSimulation", "TeleoperationController", "ALOHAEnvironment"]
        
        for class_name in classes_to_test:
            try:
                cls = getattr(robot_arm_teleop, class_name)
                print(f"✓ {class_name} accessible")
            except ImportError as e:
                print(f"⚠ {class_name} import failed (missing dependencies): {e}")
            except Exception as e:
                print(f"✗ {class_name} access failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Class access test failed: {e}")
        return False

def test_package_structure():
    """Test package directory structure."""
    print("\\nTesting package structure...")
    
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    robot_arm_teleop_dir = os.path.join(package_root, "robot_arm_teleop")
    
    required_modules = [
        "simulation",
        "teleoperation", 
        "environments",
        "utils"
    ]
    
    all_good = True
    for module in required_modules:
        module_path = os.path.join(robot_arm_teleop_dir, module)
        if os.path.isdir(module_path):
            init_file = os.path.join(module_path, "__init__.py")
            if os.path.isfile(init_file):
                print(f"✓ Module {module} exists with __init__.py")
            else:
                print(f"✗ Module {module} missing __init__.py")
                all_good = False
        else:
            print(f"✗ Module {module} directory not found")
            all_good = False
    
    # Check for examples and tests
    examples_dir = os.path.join(package_root, "examples")
    tests_dir = os.path.join(package_root, "tests")
    
    if os.path.isdir(examples_dir):
        print("✓ Examples directory exists")
    else:
        print("⚠ Examples directory not found")
    
    if os.path.isdir(tests_dir):
        print("✓ Tests directory exists")
    else:
        print("⚠ Tests directory not found")
    
    return all_good

def test_requirements_file():
    """Test that requirements.txt exists and is readable."""
    print("\\nTesting requirements file...")
    
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_file = os.path.join(package_root, "requirements.txt")
    
    if os.path.isfile(requirements_file):
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.read().strip().split('\\n')
            
            print("✓ requirements.txt exists")
            print(f"✓ Found {len(requirements)} requirements:")
            for req in requirements:
                if req.strip():
                    print(f"  - {req.strip()}")
            return True
        except Exception as e:
            print(f"✗ Error reading requirements.txt: {e}")
            return False
    else:
        print("✗ requirements.txt not found")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("ROBOT ARM TELEOPERATION PACKAGE TEST")
    print("="*60)
    
    tests = [
        test_basic_imports,
        test_class_access,
        test_package_structure,
        test_requirements_file
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("✓ All tests passed! Package structure is correct.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())