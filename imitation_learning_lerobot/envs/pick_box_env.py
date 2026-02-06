import os
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env

from ..arm.robot import Robot, UR5e
from ..arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from ..utils import mj


class PickBoxEnv(Env):
    _name = "pick_box"
    _robot_type = "UR5e"
    _height = 480
    _width = 640
    _states = [
        "px",
        "py",
        "pz",
        "gripper"
    ]
    _cameras = [
        "top",
        "hand"
    ]
    _state_dim = 4
    _action_dim = 4

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500
        self._control_hz = 25

        self._render_mode = render_mode

        self._latest_action = None
        self._render_cache = None

        scene_path = Path(__file__).parent.parent / Path("assets/scenes/pick_box_scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot: Robot = UR5e()
        self._robot_q = np.zeros(self._robot.dof)
        self._ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                  "wrist_2_joint", "wrist_3_joint"]
        self._robot_T = sm.SE3()
        self._T0 = sm.SE3()

        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        self._obj_t = np.zeros(3)
        
                # ... 其他代码 ...
        # 只创建一次，作为类成员变量
        self._jacp = np.zeros((3, self._mj_model.nv))
        self._jacr = np.zeros((3, self._mj_model.nv))


    def reset(self):

        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.disable_base()
        self._robot.disable_tool()

        self._robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "ur5e_base"))
        self._robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._robot.set_joint(self._robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._robot_q[i]) for i, jn in
         enumerate(self._ur5e_joint_names)]
        mujoco.mj_forward(self._mj_model, self._mj_data)
        mj.attach(self._mj_model, self._mj_data, "attach", "2f85", self._robot.fkine(self._robot_q))
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.15)) 
        self._robot_T = self._robot.fkine(self._robot_q)
        self._T0 = self._robot_T.copy()

        px_box = np.random.uniform(low=1.4, high=1.5)
        py_box = np.random.uniform(low=0.3, high=0.9)
        pz_box = 0.77
        T_Box = sm.SE3.Trans(px_box, py_box, pz_box)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        px_container = np.random.uniform(low=1.4, high=1.5)
        py_container = np.random.uniform(low=0.3, high=0.9)
        pz_container = 0.77
        while np.linalg.norm(
                np.array([px_box, py_box, pz_box] - np.array([px_container, py_container, pz_container]))) < 0.2:
            px_container = np.random.uniform(low=1.4, high=1.5)
            py_container = np.random.uniform(low=0.3, high=0.9)
            pz_container = 0.77
        T_container = sm.SE3.Trans(px_container, py_container, pz_container)

        container_eq_data = np.zeros(11)
        container_eq_data[3:6] = T_container.t
        container_eq_data[6:10] = T_container.UnitQuaternion()
        container_eq_data[-1] = 1.0
        mj.attach(self._mj_model, self._mj_data, "container_attach",
                  "container_free_joint", T_container, eq_data=container_eq_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human":
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data,
                                                           show_left_ui=False, show_right_ui=False)

        self._step_num = 0
        observation = self._get_observation()
        info = {"is_success": False}
        return observation, info
    

    def step(self, action):
        n_steps = self._sim_hz // self._control_hz
        if action is not None:
            self._latest_action = action

            Ti = self._T0 * sm.SE3.Trans(action[0], action[1], action[2])
            print("Action: X={:.6f}, Y={:.6f}, Z={:.6f}, Gripper={:.3f}".format(action[0], action[1], action[2], action[3]))
            print("self._T0  = \n", self._T0 ) 
            print("Ti = \n", Ti )
            print ("-----------------------------------------")
            self._robot.move_cartesian(Ti)
            joint_position = self._robot.get_joint()

            # --- 开始：奇异性检测 ---
            # 计算雅可比矩阵（针对末端执行器，如夹爪 "2f85"）
            # 注意：需要在设置控制器目标之前，但在计算出 joint_position 之后进行
            # 使用 'wrist_3_link' 作为末端执行器进行计算，因为它代表了机器人末端
            ee_site_name = 'wrist_3_link'
            mujoco.mj_jac(self._mj_model, self._mj_data,self._jacp, self._jacr, self._mj_data.body(ee_site_name).xpos, self._mj_model.body(ee_site_name).id)
            J = self._jacp[:3, :6]  # 取位置部分的前 6 关节

            # 计算条件数
            cond_num = np.linalg.cond(J)
            print()

            # 检测奇异性
            singularity_threshold = 1e3  # 可调整阈值
            if cond_num > singularity_threshold:
                print(f"⚠️  检测到奇异性，条件数: {cond_num:.2e}，重置到初始姿态")
                # 重置逻辑：将目标关节位置设为初始安全位置
                joint_position = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
                # 同时更新内部机器人对象的状态以保持同步
                self._robot.set_joint(joint_position)
            # --- 结束：奇异性检测 ---

            self._mj_data.ctrl[:6] = joint_position
            action[3] = np.clip(action[3], 0, 1)
            self._mj_data.ctrl[6] = action[3] * 255.0
        mujoco.mj_step(self._mj_model, self._mj_data, n_steps)

        observation = self._get_observation()
        reward = 0.0
        terminated = False

        self._step_num += 1

        truncated = False
        if self._step_num > 10000:
            truncated = True

        info = {"is_success": terminated}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._render_mode == "human":
            self._mj_viewer.sync()

    def close(self):
        if self._mj_viewer is not None:
            self._mj_viewer.close()
        if self._mj_renderer is not None:
            self._mj_renderer.close()

    def seed(self, seed=None):
        pass

    def _get_observation(self):
        mujoco.mj_forward(self._mj_model, self._mj_data)

        for i in range(len(self._ur5e_joint_names)):
            self._robot_q[i] = mj.get_joint_q(self._mj_model, self._mj_data, self._ur5e_joint_names[i])[0]
        self._robot_T = self._T0.inv() * self._robot.fkine(self._robot_q)
        agent_pos = np.zeros(4, dtype=np.float32)
        agent_pos[:3] = self._robot_T.t
        agent_pos[3] = np.linalg.norm(self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos)

        self._mj_renderer.update_scene(self._mj_data, 0)
        image_top = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, 1)
        image_hand = self._mj_renderer.render()

        obs = {
            'pixels': {
                'top': image_top,
                'hand': image_hand
            },
            'agent_pos': agent_pos
        }
        self._render_cache = image_top
        return obs


def main():
    """简单的测试主函数，用于交互式测试环境"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 PickBoxEnv 环境")
    parser.add_argument("--mode", type=str, default="interactive", 
                       choices=["interactive", "random", "keyboard"],
                       help="控制模式: interactive(手动输入), random(随机动作), keyboard(键盘控制)")
    parser.add_argument("--steps", type=int, default=1000, 
                       help="运行的步数")
    args = parser.parse_args()
    
    # 创建环境
    print("创建环境...")
    env = PickBoxEnv(render_mode="human")
    
    # 重置环境
    print("重置环境...")
    observation, info = env.reset()
    print(f"初始观察: agent_pos={observation['agent_pos']}")
    print(f"观察图像形状: top={observation['pixels']['top'].shape}, hand={observation['pixels']['hand'].shape}")
    
    step_count = 0
    action = np.zeros(4)  # [dx, dy, dz, gripper]
    
    print("\n" + "="*60)
    print("控制说明:")
    print("  Action 格式: [dx, dy, dz, gripper]")
    print("    - dx, dy, dz: 末端执行器的相对位置变化")
    print("    - gripper: 夹爪开合 (0.0=关闭, 1.0=打开)")
    print("="*60)
    
    if args.mode == "keyboard":
        print("\n键盘控制模式:")
        print("  使用 WASD + QE 控制移动")
        print("  W/S: Y轴前后")
        print("  A/D: X轴左右")
        print("  Q/E: Z轴上下")
        print("  Space: 打开夹爪")
        print("  Shift: 关闭夹爪")
        print("  R: 重置环境")
        print("  ESC: 退出")
        
        try:
            import keyboard as kb
            kb_available = True
        except ImportError:
            print("警告: keyboard 库未安装，无法使用键盘控制模式")
            print("请安装: pip install keyboard")
            kb_available = False
            args.mode = "interactive"
    
    try:
        while step_count < args.steps:
            if args.mode == "random":
                # 随机动作
                action = np.random.uniform(-0.01, 0.01, size=3).tolist() + [np.random.choice([0.0, 1.0])]
                action = np.array(action)
                
            elif args.mode == "keyboard" and kb_available:
                # 键盘控制
                action[:3] = 0.0  # 重置位置增量
                action[3] = 0.5  # 默认夹爪位置
                
                if kb.is_pressed('w'):
                    action[1] += 0.001  # Y轴向前
                if kb.is_pressed('s'):
                    action[1] -= 0.001  # Y轴向后
                if kb.is_pressed('a'):
                    action[0] -= 0.001  # X轴向左
                if kb.is_pressed('d'):
                    action[0] += 0.001  # X轴向右
                if kb.is_pressed('q'):
                    action[2] += 0.001  # Z轴向上
                if kb.is_pressed('e'):
                    action[2] -= 0.001  # Z轴向下
                if kb.is_pressed('space'):
                    action[3] = 1.0  # 打开夹爪
                if kb.is_pressed('shift'):
                    action[3] = 0.0  # 关闭夹爪
                if kb.is_pressed('r'):
                    print("\n重置环境...")
                    observation, info = env.reset()
                    step_count = 0
                    continue
                if kb.is_pressed('esc'):
                    break
                    
            elif args.mode == "interactive":
                # 交互式输入
                if step_count % 10 == 0:  # 每10步提示一次
                    print(f"\n步骤 {step_count}/{args.steps}")
                    print(f"当前 agent_pos: {observation['agent_pos']}")
                    user_input = input("输入动作 [dx dy dz gripper] (或 'r' 重置, 'q' 退出, 回车使用上次动作): ").strip()
                    
                    if user_input.lower() == 'q':
                        break
                    elif user_input.lower() == 'r':
                        print("重置环境...")
                        observation, info = env.reset()
                        step_count = 0
                        continue
                    elif user_input:
                        try:
                            values = list(map(float, user_input.split()))
                            if len(values) == 4:
                                action = np.array(values)
                            else:
                                print("警告: 需要4个值，使用上次动作")
                        except ValueError:
                            print("警告: 输入格式错误，使用上次动作")
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 渲染
            env.render()
            
            step_count += 1
            
            # 检查是否结束
            if terminated or truncated:
                print(f"\nEpisode 结束: terminated={terminated}, truncated={truncated}")
                observation, info = env.reset()
                step_count = 0
            
            # 控制频率
            time.sleep(1.0 / env._control_hz)
            
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在退出...")
    finally:
        print(f"\n总共执行了 {step_count} 步")
        print("关闭环境...")
        env.close()
        print("完成!")


if __name__ == "__main__":
    main()
