import argparse
import time
import numpy as np

# Make sure the environment is importable
from imitation_learning_lerobot.envs.pick_and_place_env import PickAndPlaceEnv

# Attempt to import the keyboard library for keyboard mode
try:
    import keyboard as kb
    KB_AVAILABLE = True
except ImportError:
    print("Warning: 'keyboard' library not found. Keyboard mode will be disabled.")
    print("Install it with: pip install keyboard")
    KB_AVAILABLE = False

def main():
    """A script to interactively control the PickAndPlaceEnv."""
    parser = argparse.ArgumentParser(description="Play with the PickAndPlaceEnv")
    parser.add_argument(
        "--mode",
        type=str,
        default="keyboard",
        choices=["keyboard", "interactive", "random"],
        help="Control mode: 'keyboard' for real-time control, 'interactive' for command input, 'random' for random actions."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Maximum number of steps to run the simulation."
    )
    args = parser.parse_args()

    if args.mode == "keyboard" and not KB_AVAILABLE:
        print("Keyboard mode selected but library is not available. Switching to 'interactive' mode.")
        args.mode = "interactive"

    # --- Setup ---
    env = PickAndPlaceEnv(render_mode="human")
    observation, info = env.reset()
    
    # For absolute control, we need to start from the robot's initial position
    # The observation['agent_pos'] contains the absolute [x, y, z, gripper_width]
    current_target_pose = observation['agent_pos'].copy()
    action = current_target_pose.copy()

    step_count = 0
    
    # --- Print Instructions ---
    print("\n" + "="*60)
    print("Controlling PickAndPlaceEnv")
    print(f"Mode: {args.mode}")
    print("Action is an ABSOLUTE target: [x, y, z, gripper_state]")
    print("  - gripper_state: 0.0 for closed, 1.0 for open")
    
    if args.mode == "keyboard":
        print("\n--- Keyboard Controls ---")
        print("  W/S: Move target in +Y / -Y")
        print("  A/D: Move target in -X / +X")
        print("  Q/E: Move target in +Z / -Z")
        print("  Space: Open gripper (set to 1.0)")
        print("  Shift: Close gripper (set to 0.0)")
        print("  R: Reset environment")
        print("  ESC: Exit")
    print("="*60 + "\n")

    # --- Main Loop ---
    try:
        while step_count < args.steps:
            # --- Determine Action Based on Mode ---
            if args.mode == "keyboard":
                # Incremental changes to the absolute target pose
                step_size = 0.01
                if kb.is_pressed('w'): current_target_pose[1] += step_size
                if kb.is_pressed('s'): current_target_pose[1] -= step_size
                if kb.is_pressed('a'): current_target_pose[0] -= step_size
                if kb.is_pressed('d'): current_target_pose[0] += step_size
                if kb.is_pressed('q'): current_target_pose[2] += step_size
                if kb.is_pressed('e'): current_target_pose[2] -= step_size
                
                if kb.is_pressed('space'): current_target_pose[3] = 1.0
                if kb.is_pressed('shift'): current_target_pose[3] = 0.0

                if kb.is_pressed('r'):
                    print("Resetting environment...")
                    observation, info = env.reset()
                    current_target_pose = observation['agent_pos'].copy()
                    step_count = 0
                    continue

                if kb.is_pressed('esc'):
                    print("Exiting...")
                    break
                
                action[:] = current_target_pose

            elif args.mode == "interactive":
                print(f"\nStep {step_count}/{args.steps} | Current Pose: {observation['agent_pos']}")
                user_input = input("Enter target pose [x y z gripper] (or 'r' to reset, 'q' to quit): ").strip()

                if user_input.lower() == 'q': break
                if user_input.lower() == 'r':
                    observation, info = env.reset()
                    current_target_pose = observation['agent_pos'].copy()
                    step_count = 0
                    continue
                
                try:
                    values = list(map(float, user_input.split()))
                    if len(values) == 4:
                        action = np.array(values, dtype=np.float32)
                        current_target_pose = action.copy()
                    else:
                        print("Invalid input. Expected 4 numbers. Using previous action.")
                except ValueError:
                    print("Parse error. Using previous action.")

            elif args.mode == "random":
                # Generate a random target within a reasonable workspace
                action[0] = np.random.uniform(1.3, 1.6)
                action[1] = np.random.uniform(0.2, 1.0)
                action[2] = np.random.uniform(0.78, 0.9)
                action[3] = np.random.choice([0.0, 1.0])
                print(f"Random Action: {action}")

            # --- Step the Environment ---
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                print(f"Episode finished. Resetting.")
                observation, info = env.reset()
                current_target_pose = observation['agent_pos'].copy()
                step_count = 0

            step_count += 1
            time.sleep(1.0 / env._control_hz)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Closing environment.")
        env.close()

if __name__ == "__main__":
    main()
