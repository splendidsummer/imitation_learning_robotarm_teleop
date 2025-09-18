from typing import Type
from pathlib import Path
import argparse
import pygame
import threading
import time

from loop_rate_limiters import RateLimiter
import numpy as np
import h5py
import cv2

from imitation_learning_lerobot.envs import Env, EnvFactory


# Direct joystick handler based on joystick_control.py logic
class DirectJoystickHandler:
    def __init__(self):
        self._timestep = 0.01
        self._action = np.zeros(4)
        
        # Initialize pygame joystick
        pygame.init()
        pygame.joystick.init()
        
        self._joystick = None
        if pygame.joystick.get_count() > 0:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            print(f"Xbox Controller detected: {self._joystick.get_name()}")
        else:
            print("Warning: No Xbox controller detected!")

        self._joystick_calibration_offset = np.zeros(2)  # Only need 2D for joystick calibration
        self._deadzone = 0.1
        
        # Control parameters from joystick_control.py
        self._translation_scale = 0.002
        self._rotation_scale = 0.5
        self._gripper_delta = 0.005

        self._sync = False
        self._done = False
        self._running = True
        self._thread = None

    def _calibrate(self):
        if self._joystick is None:
            print("No joystick available for calibration")
            return
            
        print("Calibrating Xbox controller... Keep joysticks centered.")
        num_samples = 100
        joystick_samples = []
        
        for _ in range(num_samples):
            pygame.event.pump()  # Update joystick state
            
            # Read left joystick axes (assuming left stick for movement)
            left_x = self._joystick.get_axis(0) if self._joystick.get_numaxes() > 0 else 0.0
            left_y = self._joystick.get_axis(1) if self._joystick.get_numaxes() > 1 else 0.0
            
            joystick_samples.append([left_x, left_y])
            time.sleep(0.01)

        self._joystick_calibration_offset[:] = np.mean(joystick_samples, axis=0)
        print(f"Calibration complete. Offset: {self._joystick_calibration_offset}")

    def start(self):
        time.sleep(1.0)
        self._calibrate()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        rate_limiter = RateLimiter(frequency=1.0 / self._timestep)
        while self._running:
            self._xbox_update()
            rate_limiter.sleep()

    def _apply_deadzone(self, value):
        """Apply deadzone to joystick input"""
        return value if abs(value) >= self._deadzone else 0.0

    def _xbox_update(self):
        if self._joystick is None:
            return
            
        pygame.event.pump()  # Update joystick state
        
        # Sync control: A button to start, Y button to pause
        if not self._sync:
            if self._joystick.get_numbuttons() > 0 and self._joystick.get_button(0):  # A button
                self._sync = True
                print("Started recording")
        else:
            if self._joystick.get_numbuttons() > 3 and self._joystick.get_button(3):  # Y button
                self._sync = False
                print("Paused recording")
                
        # Stop recording: Back/Select button (button 6 on Xbox controller)
        if self._joystick.get_numbuttons() > 6 and self._joystick.get_button(6):
            self._done = True
            print("Stopping recording")

        if not self._sync:
            return

        # Read joystick axes with calibration and deadzone
        left_x = self._joystick.get_axis(0) if self._joystick.get_numaxes() > 0 else 0.0
        left_y = self._joystick.get_axis(1) if self._joystick.get_numaxes() > 1 else 0.0
        
        # Apply calibration offset
        left_x_cal = left_x - self._joystick_calibration_offset[0]
        left_y_cal = left_y - self._joystick_calibration_offset[1]
        
        # Apply deadzone
        left_x_cal = self._apply_deadzone(left_x_cal)
        left_y_cal = self._apply_deadzone(left_y_cal)
        
        # Read buttons for Z-axis and gripper control
        # Left trigger (LT) for down, Right trigger (RT) for up
        left_trigger = self._joystick.get_axis(2) if self._joystick.get_numaxes() > 2 else 0.0  # LT
        right_trigger = self._joystick.get_axis(5) if self._joystick.get_numaxes() > 5 else 0.0  # RT
        
        # Convert triggers from [-1, 1] to [0, 1] range
        left_trigger = (left_trigger + 1) / 2
        right_trigger = (right_trigger + 1) / 2
        
        # Gripper control: X button to close, B button to open
        close_gripper = self._joystick.get_button(2) if self._joystick.get_numbuttons() > 2 else 0  # X button
        open_gripper = self._joystick.get_button(1) if self._joystick.get_numbuttons() > 1 else 0  # B button

        # Update action array (same logic as original but with Xbox controller inputs)
        self._action[0] -= left_x_cal * 0.000002  # X axis (left/right)
        self._action[1] += left_y_cal * 0.000002  # Y axis (forward/back) - inverted
        self._action[2] += 0.002 if right_trigger > 0.5 else -0.002 if left_trigger > 0.5 else 0  # Z axis
        self._action[3] += 0.01 if open_gripper == 1 else -0.01 if close_gripper == 1 else 0.0  # Gripper
        self._action[3] = np.clip(self._action[3], 0.0, 1.0)

    @property
    def action(self):
        return self._action.copy()

    @property
    def sync(self):
        return self._sync

    @property
    def done(self):
        return self._done

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join()
        if self._joystick:
            self._joystick.quit()
        pygame.quit()

    def print_info(self):
        print("------------------------------")
        print("Xbox Controller Mapping:")
        print("Start:           A")
        print("Pause:           Y")
        print("Stop:            Back/Select")
        print("Movement:        Left Joystick")
        print("+Z (Up):         Right Trigger (RT)")
        print("-Z (Down):       Left Trigger (LT)")
        print("Open Gripper:    B")
        print("Close Gripper:   X")
        print("------------------------------")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env.type',
        type=str,
        dest='env_type',
        required=True,
        help='env type'
    )

    parser.add_argument(
        '--handler.type',
        type=str,
        dest='handler_type',
        default='joystick',
        help='handler type (default: joystick)'
    )

    return parser.parse_args()


def teleoperate(env_cls: Type[Env], handler_type):
    # Use direct joystick handler instead of factory pattern
    if handler_type == 'joystick' or handler_type == 'joycon':
        handler = DirectJoystickHandler()
    else:
        raise ValueError(f"Unsupported handler type: {handler_type}")
    
    handler.start()
    handler.print_info()

    env = env_cls(render_mode="human")
    observation, info = env.reset()

    for camera in env_cls.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    data_dict = {
        '/observations/agent_pos': [],
        **{f'/observations/pixels/{camera}': [] for camera in env_cls.cameras},
        '/actions': []
    }

    rate_limiter = RateLimiter(frequency=env.control_hz)

    action = handler.action
    last_action = action.copy()
    
    print("Waiting for controller input... Press A to start recording.")
    
    while not handler.done:
        if not handler.sync:
            rate_limiter.sleep()
            continue

        last_action[:] = action
        action[:] = handler.action
        if np.max(np.abs(action - last_action)) > 1e-6:
            data_dict['/observations/agent_pos'].append(observation['agent_pos'])
            for camera in env_cls.cameras:
                data_dict[f'/observations/pixels/{camera}'].append(observation['pixels'][camera])
            data_dict['/actions'].append(action.copy())
        else:
            action[:] = last_action

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()
        for camera in env.cameras:
            cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        rate_limiter.sleep()

    cv2.destroyAllWindows()
    handler.close()
    env.close()

    print(f"Recording completed. Collected {len(data_dict['/actions'])} samples.")
    return data_dict


def write_to_h5(env_cls: Type[Env], data_dict: dict):
    h5_dir = Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name + "_hdf5")
    h5_dir.mkdir(parents=True, exist_ok=True)

    index = len([f for f in h5_dir.iterdir() if f.is_file()])
    h5_path = h5_dir / Path(f"episode_{index:06d}.hdf5")

    print(f"Saving data to: {h5_path}")
    print(f"Episode length: {len(data_dict['/actions'])} steps")

    with h5py.File(h5_path, 'w') as root:
        episode_length = len(data_dict['/actions'])

        obs = root.create_group('observations')
        obs.create_dataset('agent_pos', (episode_length, env_cls.state_dim), dtype='float32', compression='gzip')

        pixels = obs.create_group('pixels')
        for camera in env_cls.cameras:
            shape = (episode_length, env_cls.height, env_cls.width, 3)
            chunks = (1, env_cls.height, env_cls.width, 3)
            pixels.create_dataset(camera, shape=shape, dtype='uint8', chunks=chunks, compression='gzip')

        root.create_dataset('actions', (episode_length, env_cls.action_dim), dtype='float32', compression='gzip')

        for name, array in data_dict.items():
            root[name][...] = array
            
    print(f"Successfully saved episode {index:06d} with {episode_length} steps")


def main():
    args = parse_args()

    try:
        env_cls = EnvFactory.get_strategies(args.env_type)
        print(f"Using environment: {env_cls.name}")
        print(f"Using handler: {args.handler_type}")
        
        data_dict = teleoperate(env_cls, args.handler_type)
        
        if len(data_dict['/actions']) > 0:
            write_to_h5(env_cls, data_dict)
            print("Data successfully saved!")
        else:
            print("No data collected. Make sure to press A to start recording and move the controller.")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == '__main__':
    main()
