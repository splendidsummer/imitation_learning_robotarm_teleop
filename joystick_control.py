import mujoco
import mujoco.viewer
import pygame
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------- 配置 ----------
USE_GUI = True  # False 用无图形模式
CONTROL_RATE_HZ = 240
TRANSLATION_SCALE = 0.002  # 操纵杆映射到末端平移速度
ROTATION_SCALE = 0.5  # 操纵杆映射到末端旋转速度（弧度/秒）
GRIPPER_DELTA = 0.005  # 抓手开合步进
DEADZONE = 0.1  # 操纵杆死区
# -------------------------

# 加载 MuJoCo 模型
MODEL_PATH = "./franka_emika_panda/panda.xml"  # 替换为你的模型路径


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
print(f"成功加载模型: {MODEL_PATH}")

# 初始化查看器
if USE_GUI:
    viewer = mujoco.viewer.launch_passive(model, data)
else:
    viewer = None

# 获取关节、体和执行器的索引
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

# 创建名称到索引的映射
joint_name_to_index = {name: i for i, name in enumerate(joint_names) if name}
body_name_to_index = {name: i for i, name in enumerate(body_names) if name}
actuator_name_to_index = {name: i for i, name in enumerate(actuator_names) if name}

print("可用的关节:", [name for name in joint_names if name])
print("可用的执行器:", [name for name in actuator_names if name])
print("可用的体:", [name for name in body_names if name])

# 末端链接和抓手关节索引
ee_body_name = "panda_hand"
ee_body_id = body_name_to_index.get(ee_body_name, 2)  # 默认为2（简单模型中的手部）

# 抓手执行器索引（使用执行器而不是关节）
gripper_actuator_left = actuator_name_to_index.get("gripper_left", None)
gripper_actuator_right = actuator_name_to_index.get("gripper_right", None)

# 主关节执行器索引
joint1_actuator = actuator_name_to_index.get("motor1", 0)
joint2_actuator = actuator_name_to_index.get("motor2", 1)

# 初始抓手开度
gripper_opening = 0.04  # max ~0.04

def set_gripper(width):
    # MuJoCo 抓手控制 - 使用执行器索引
    target = max(0.0, min(0.04, width / 2.0))
    if gripper_actuator_left is not None and gripper_actuator_left < model.nu:
        data.ctrl[gripper_actuator_left] = target
    if gripper_actuator_right is not None and gripper_actuator_right < model.nu:
        data.ctrl[gripper_actuator_right] = target

set_gripper(gripper_opening)

# 初始化 joystick
pygame.init()
pygame.joystick.init()
joy = None
if pygame.joystick.get_count() == 0:
    print("警告：未检测到操纵杆，脚本将运行但没有输入。")
else:
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"使用操纵杆: {joy.get_name()}, 轴数: {joy.get_numaxes()}, 按钮数: {joy.get_numbuttons()}")

# 读取当前末端位姿作为起点
def get_end_effector_pose():
    # 更新正运动学
    mujoco.mj_forward(model, data)
    
    # 获取末端体的位置和方向
    pos = data.xpos[ee_body_id].copy()
    mat = data.xmat[ee_body_id].reshape(3, 3)
    
    # 将旋转矩阵转换为四元数
    r = R.from_matrix(mat)
    quat = r.as_quat()  # [x, y, z, w]
    
    return pos, quat

# 主控制循环
target_pos, target_ori = get_end_effector_pose()
use_rotation_mode = False  # False 平移，True 旋转

print("按钮映射示意（以一般 Xbox 风格为例）：")
print("  A/B/X/Y 等可用于模式切换、抓手开合，具体看下面注释里的按钮编号。")
print("  按 Ctrl+C 退出程序")
print("开始控制循环...")

try:
    while True:
        t_start = time.time()
        pygame.event.pump()  # 更新操纵杆状态
        
        # 检查pygame事件，处理退出
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        # 读操纵杆轴
        if joy is not None and pygame.joystick.get_count() > 0:
            # 左摇杆： axes[0], axes[1] 控制平移 X/Y
            # 右摇杆： axes[3], axes[4] 控制平移 Z 和 旋转（举例）
            ax_x = joy.get_axis(0) if joy.get_numaxes() > 0 else 0.0
            ax_y = joy.get_axis(1) if joy.get_numaxes() > 1 else 0.0
            ax_z = joy.get_axis(3) if joy.get_numaxes() > 3 else 0.0
            ax_rot = joy.get_axis(4) if joy.get_numaxes() > 4 else 0.0

            # 死区
            def apply_deadzone(v):
                return v if abs(v) >= DEADZONE else 0.0

            dx = apply_deadzone(ax_x) * TRANSLATION_SCALE
            dy = apply_deadzone(-ax_y) * TRANSLATION_SCALE  # 反向倒置习惯
            dz = apply_deadzone(-ax_z) * TRANSLATION_SCALE

            drot = apply_deadzone(ax_rot) * ROTATION_SCALE

            # 模式切换按钮（假设按钮 0 切换平移/旋转, 按钮 1 合上抓手, 按钮 2 张开抓手）
            if joy.get_numbuttons() >= 1 and joy.get_button(0):
                use_rotation_mode = not use_rotation_mode
                print(f"切换模式: {'旋转模式' if use_rotation_mode else '平移模式'}")
                time.sleep(0.2)  # 简单防抖

            # 抓手控制
            if joy.get_numbuttons() >= 2 and joy.get_button(1):  # 关闭
                gripper_opening = max(0.0, gripper_opening - GRIPPER_DELTA)
                set_gripper(gripper_opening)
            if joy.get_numbuttons() >= 3 and joy.get_button(2):  # 打开
                gripper_opening = min(0.04, gripper_opening + GRIPPER_DELTA)
                set_gripper(gripper_opening)

            # 末端控制
            if use_rotation_mode:
                # 使用scipy的Rotation来处理旋转
                r = R.from_quat(target_ori)
                euler = r.as_euler('xyz')
                euler[2] += drot  # 绕 Z 旋转
                target_ori = R.from_euler('xyz', euler).as_quat()
            else:
                delta = np.array([dx, dy, dz])
                target_pos = target_pos + delta

        # MuJoCo 中使用正向运动学和简化控制
        # 注意: MuJoCo 没有内置的逆运动学求解器，这里使用简化的关节控制
        # 实际应用中可能需要集成外部IK库或使用数值方法
        
        # 简化的关节控制 - 使用执行器索引
        if model.nu >= 2:
            # 简单的末端位置到关节角度的映射（这是简化版本）
            data.ctrl[joint1_actuator] = np.clip(target_pos[0] * 2, -2.8, 2.8)  # Joint 1
            data.ctrl[joint2_actuator] = np.clip(target_pos[2] - 0.4, -1.7, 1.7)  # Joint 2

        # 执行仿真步进
        mujoco.mj_step(model, data)
        
        # 更新可视化
        if viewer is not None:
            viewer.sync()

        # 维持频率
        elapsed = time.time() - t_start
        sleep_t = max(0, (1.0 / CONTROL_RATE_HZ) - elapsed)
        time.sleep(sleep_t)

except KeyboardInterrupt:
    print("\n接收到中断信号，正在退出...")
except Exception as e:
    print(f"\n发生错误: {e}")
finally:
    # 清理资源
    if viewer is not None:
        viewer.close()
    pygame.quit()
    print("程序已退出")
