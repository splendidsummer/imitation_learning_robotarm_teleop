# imitation_learning_lerobot

## 介绍
imitation_learning lerobot is used to do the teleoperation of robot arm with the pick-and-place task. For the controller we currently support **joystick** and **joy-con R**. And the **keyboard** is suppoorted for the default controller. 

## Running Teleop Control 

### Building the Python Env 
```bash
conda create -n  imitation_learning python==3.10.0 
conda activate imitation_learning

pip install -r requirements.txt

``` 

### Using JoyCon R
```bash 
conda activate imitation_learning
 
python -m imitation_learning_lerobot.scripts.collect_data_teleoperation --env.type=pick_box  --handler.type=joycon_evdev
```

### Using JoyStick 
```bash
conda activate imitation_learning

python -m imitation_learning_lerobot.scripts.collect_data_joystick_teleoperation --env.type=pick_box
``` 

### Using Keyboard

```bash
conda activate imitation_learning
 
python -m imitation_learning_lerobot.scripts.collect_data_teleoperation --env.type=pick_box  --handler.type=keyboard
```
Below is the keyboard button to control the robot arm. 
```
------------------------------
Start:           Right Ctrl
Pause:           Right Shift
Stop:            Enter
+X:              Keypad 1
-X:              Keypad 7
+Y:              Keypad 6
-Y:              Keypad 4
+Z:              Keypad 8
-Z:              Keypad 2
Open:            Keypad 3
Close:           Keypad 9
The environment cameras including: top, hand
```