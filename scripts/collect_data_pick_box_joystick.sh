#!/bin/bash

# Collect data using Xbox joystick for pick_box environment
python ./imitation_learning_lerobot/scripts/collect_data_joystick_teleopration.py \
  --env.type=pick_box \
  --handler.type=joystick
