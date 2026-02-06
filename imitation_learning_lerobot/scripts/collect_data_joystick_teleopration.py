from typing import Type, List
from pathlib import Path
import argparse
import pygame
import time

from loop_rate_limiters import RateLimiter
import numpy as np
import h5py
import cv2

from imitation_learning_lerobot.envs import Env, EnvFactory


# Direct joystick handler - Single Threaded Version
class DirectJoystickHandler:
    def __init__(self):
        self._timestep = 0.02  # 50 Hz
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

        self._joystick_calibration_offset = np.zeros(2)
        self._deadzone = 0.1
        
        self._translation_scale = 0.002
        self._rotation_scale = 0.5
        self._gripper_delta = 0.005

        self._sync = False
        self._done = False
        self._reset_episode = False
        self._reset_latch = False
        self._running = True

    def _calibrate(self):
        if self._joystick is None:
            print("No joystick available for calibration")
            return
            
        print("Calibrating Xbox controller... Keep joysticks centered.")
        num_samples = 50 
        joystick_samples = []
        
        for _ in range(num_samples):
            pygame.event.pump()
            left_x = self._joystick.get_axis(0) if self._joystick.get_numaxes() > 0 else 0.0
            left_y = self._joystick.get_axis(1) if self._joystick.get_numaxes() > 1 else 0.0
            joystick_samples.append([left_x, left_y])
            time.sleep(0.01)

        self._joystick_calibration_offset[:] = np.mean(joystick_samples, axis=0)
        print(f"Calibration complete. Offset: {self._joystick_calibration_offset}")

    def start(self):
        time.sleep(0.5)
        self._calibrate()

    def _apply_deadzone(self, value):
        return value if abs(value) >= self._deadzone else 0.0

    def update(self):
        if self._joystick is None:
            return
            
        pygame.event.pump()
        
        # Sync control: A button (0) to start, Y button (3) to pause
        if not self._sync:
            if self._joystick.get_numbuttons() > 0 and self._joystick.get_button(0):
                self._sync = True
                print("Started recording")
        else:
            if self._joystick.get_numbuttons() > 3 and self._joystick.get_button(3):
                self._sync = False
                print("Paused recording")
                
        # Stop session: Back button (6)
        if self._joystick.get_numbuttons() > 6 and self._joystick.get_button(6):
            self._done = True
            print("Stopping session")

        # Reset Episode: Start button (7) with Latch
        is_reset_pressed = self._joystick.get_numbuttons() > 7 and self._joystick.get_button(7)
        
        if is_reset_pressed:
            if not self._reset_latch:
                self._reset_episode = True
                self._reset_latch = True
                print("Resetting episode...")
        else:
            self._reset_latch = False

        if not self._sync:
            return

        # Axes
        left_x = self._joystick.get_axis(0) if self._joystick.get_numaxes() > 0 else 0.0
        left_y = self._joystick.get_axis(1) if self._joystick.get_numaxes() > 1 else 0.0
        
        left_x_cal = self._apply_deadzone(left_x - self._joystick_calibration_offset[0])
        left_y_cal = self._apply_deadzone(left_y - self._joystick_calibration_offset[1])
        
        left_trigger = (self._joystick.get_axis(2) + 1) / 2 if self._joystick.get_numaxes() > 2 else 0.0
        right_trigger = (self._joystick.get_axis(5) + 1) / 2 if self._joystick.get_numaxes() > 5 else 0.0
        
        close_gripper = self._joystick.get_button(2) if self._joystick.get_numbuttons() > 2 else 0
        open_gripper = self._joystick.get_button(1) if self._joystick.get_numbuttons() > 1 else 0

        # Update action
        self._action[0] -= left_x_cal * self._translation_scale
        self._action[1] += left_y_cal * self._translation_scale
        self._action[2] += 0.002 if right_trigger > 0.5 else -0.002 if left_trigger > 0.5 else 0
        self._action[3] += 0.01 if open_gripper == 1 else -0.01 if close_gripper == 1 else 0.0
        self._action[3] = np.clip(self._action[3], 0.0, 1.0)

        # Workspace Limits
        current_pos = self._action[:3]
        distance = np.linalg.norm(current_pos)
        max_reach = 0.45 
        if distance > max_reach:
            self._action[:3] = (current_pos / distance) * max_reach
        if self._action[2] < 0.02:
            self._action[2] = 0.02

    @property
    def action(self):
        return self._action.copy()

    @property
    def sync(self):
        return self._sync

    @property
    def done(self):
        return self._done

    @property
    def reset_episode(self):
        return self._reset_episode

    def ack_reset(self):
        self._reset_episode = False
        self._sync = False 

    def set_action(self, action):
        dim = min(len(action), len(self._action))
        self._action[:dim] = action[:dim]

    def close(self):
        self._running = False
        if self._joystick:
            self._joystick.quit()
        pygame.quit()

    def print_info(self):
        print("------------------------------")
        print("Xbox Controller Mapping:")
        print("Start (A):       Record")
        print("Pause (Y):       Pause")
        print("Reset (Start):   Finish Episode & Reset Arm")
        print("Quit (Back):     Save & Exit")
        print("Movement:        Left Joystick")
        print("+Z (Up):         Right Trigger (RT)")
        print("-Z (Down):       Left Trigger (LT)")
        print("Open Gripper:    B")
        print("Close Gripper:   X")
        print("------------------------------")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env.type', type=str, dest='env_type', required=True, help='env type')
    parser.add_argument('--handler.type', type=str, dest='handler_type', default='joystick', help='handler type')
    parser.add_argument('--task', type=str, dest='task', default='teleoperate robot arm', help='language instruction')
    return parser.parse_args()


def teleoperate(env_cls: Type[Env], handler_type):
    if handler_type == 'joystick' or handler_type == 'joycon':
        handler = DirectJoystickHandler()
    else:
        raise ValueError(f"Unsupported handler type: {handler_type}")
    
    handler.start()
    handler.print_info()

    # FIX: Use "human" mode to restore the main simulation window (Front View)
    env = env_cls(render_mode="human")
    
    # NOTE: We do NOT create cv2 windows here to avoid Segmentation Faults.
    # The visualization will happen in the native simulation window.

    # Initialize storage for a single episode
    current_episode = {
        '/observations/agent_pos': [],
        **{f'/observations/pixels/{camera}': [] for camera in env_cls.cameras},
        '/actions': []
    }

    print("Session started. Press 'A' to record. Press 'Back' to stop and save.")

    # Reset once at the beginning
    try:
        observation, info = env.reset()
    except Exception as e:
        print(f"Error during env.reset(): {e}")
        return []
    
    # Re-initialize RateLimiter
    rate_limiter = RateLimiter(frequency=50.0)
    
    time.sleep(0.2)
    
    if 'agent_pos' in observation:
        handler.set_action(observation['agent_pos'])

    action = handler.action
    
    print("Ready. Waiting for input...")

    # --- Single Loop (Runs until 'Back' button is pressed) ---
    while not handler.done:
        handler.update()

        # Get desired action from joystick
        desired_action = handler.action
        
        # Apply action if it changed significantly (jitter filter)
        if np.max(np.abs(desired_action - action)) > 1e-6:
            action[:] = desired_action

        # Step the environment ALWAYS (so you can see the robot move)
        observation, reward, terminated, truncated, info = env.step(action)
        
        # FIX: Explicitly render to update the simulation window
        try:
            env.render()
        except Exception:
            pass

        # Only record data if 'A' button (sync) is active
        if handler.sync:
            current_episode['/observations/agent_pos'].append(observation['agent_pos'])
            for camera in env_cls.cameras:
                # Ensure we actually have pixels to save
                if camera in observation['pixels']:
                    current_episode[f'/observations/pixels/{camera}'].append(observation['pixels'][camera].copy())
            current_episode['/actions'].append(action.copy())

        rate_limiter.sleep()

    print("Session ended.")
    
    handler.close()
    env.close()

    # Return the single episode as a list
    if len(current_episode['/actions']) > 0:
        print(f"Collected {len(current_episode['/actions'])} samples.")
        return [current_episode]
    else:
        print("No data collected.")
        return []


def write_to_h5(env_cls: Type[Env], all_episodes: List[dict], task_description: str):
    h5_dir = Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name + "_hdf5")
    h5_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    h5_path = h5_dir / Path(f"session_{timestamp}.hdf5")

    print(f"Saving {len(all_episodes)} episodes to: {h5_path}")

    with h5py.File(h5_path, 'w') as root:
        root.attrs['task'] = task_description
        
        for i, data_dict in enumerate(all_episodes):
            grp = root.create_group(f"episode_{i}")
            episode_length = len(data_dict['/actions'])

            cameras = env_cls.cameras
            wrist_cam_name = next((c for c in cameras if 'wrist' in c.lower()), None)
            main_cam_name = next((c for c in cameras if c != wrist_cam_name), cameras[0] if cameras else None)

            if main_cam_name:
                shape = (episode_length, env_cls.height, env_cls.width, 3)
                chunks = (1, env_cls.height, env_cls.width, 3)
                grp.create_dataset('image', data=np.array(data_dict[f'/observations/pixels/{main_cam_name}']), 
                                  shape=shape, dtype='uint8', chunks=chunks, compression='gzip')

            if wrist_cam_name:
                shape = (episode_length, env_cls.height, env_cls.width, 3)
                chunks = (1, env_cls.height, env_cls.width, 3)
                grp.create_dataset('wrist_image', data=np.array(data_dict[f'/observations/pixels/{wrist_cam_name}']), 
                                  shape=shape, dtype='uint8', chunks=chunks, compression='gzip')

            grp.create_dataset('state', data=np.array(data_dict['/observations/agent_pos']), 
                              dtype='float32', compression='gzip')

            grp.create_dataset('actions', data=np.array(data_dict['/actions']), 
                              dtype='float32', compression='gzip')
            
            dt = h5py.special_dtype(vlen=str)
            task_ds = grp.create_dataset('task', (episode_length,), dtype=dt)
            task_ds[:] = task_description

    print(f"Successfully saved session data.")


def main():
    args = parse_args()

    try:
        env_cls = EnvFactory.get_strategies(args.env_type)
        print(f"Using environment: {env_cls.name}")
        
        all_episodes = teleoperate(env_cls, args.handler_type)
        
        if len(all_episodes) > 0:
            write_to_h5(env_cls, all_episodes, args.task)
        else:
            print("No episodes collected.")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == '__main__':
    main()
