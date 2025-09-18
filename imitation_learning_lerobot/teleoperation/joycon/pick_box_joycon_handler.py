import threading
import time
import numpy as np
import pygame
from loop_rate_limiters import RateLimiter

import abc

import numpy as np

from ..handler import Handler


class PickBoxJoyconHandler(Handler):  
    _name = "pick_box_joycon"

    def __init__(self):
        super().__init__()

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

        self._sync = False
        self._done = False

        self._thread: threading.Thread = None
        self._running = True

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


def main():
    handler = PickBoxJoyconHandler()
    handler.print_info()
    handler.start()

    try:
        while not handler._done:
            print(f"Action: {handler._action}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        handler.close()
        print("Handler closed.")

if __name__ == "__main__":
    main()