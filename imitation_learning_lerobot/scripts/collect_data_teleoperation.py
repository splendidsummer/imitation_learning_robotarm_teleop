# 这个脚本用于通过手柄/控制器对环境进行远程操作（teleoperation），
# 并将收集到的观测（包括像素图像、agent位置）和动作保存为 HDF5 文件。
# 文件路径: /home/summer/Projects/imitation_learning_robotarm_teleop/imitation_learning_lerobot/scripts/collect_data_teleoperation.py

from typing import Type
from pathlib import Path
import argparse

from loop_rate_limiters import RateLimiter
import numpy as np
import h5py
import cv2
import math
import pickle  # 1. 导入 pickle 模块

from imitation_learning_lerobot.envs import Env, EnvFactory
from imitation_learning_lerobot.teleoperation import HandlerFactory


#  --env.type=pick_box   --handler.type=joycon
# 上面的示例表明脚本通过命令行参数指定环境类型和控制器类型

def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env.type',
        type=str,
        dest='env_type',
        required=True,
        help='env type'  # 环境类型，例如 pick_box
    )

    parser.add_argument(
        '--handler.type',
        type=str,
        dest='handler_type',
        required=True,
        help='handler type'  # 控制器/手柄类型，例如 joycon
    )

    return parser.parse_args()

#  pick_box_joycon
def teleoperate(env_cls: Type[Env], handler_type):
    # 根据环境名和 handler 类型获取具体的 handler 类，并实例化
    handler_cls = HandlerFactory.get_strategies(env_cls.name + "_" + handler_type)
    handler = handler_cls()
    handler.start()         # 启动 handler（例如开始监听手柄输入）
    handler.print_info()    # 打印 handler 信息

    # 创建环境实例并重置，获取初始观测
    env = env_cls(render_mode="human")
    observation, info = env.reset()

    print("The environment cameras including:", ", ".join(env_cls.cameras))

    # 为每个摄像头创建 OpenCV 窗口用于实时显示
    for camera in env_cls.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    # 构建用于存储数据的字典：包含 agent 位置、每个摄像头的像素以及动作序列
    data_dict = {
        '/observations/agent_pos': [],
        **{f'/observations/pixels/{camera}': [] for camera in env_cls.cameras},
        '/actions': []
    }

    # 使用限频器以与环境控制频率同步循环
    rate_limiter = RateLimiter(frequency=env.control_hz)
    max_var_action = []

    action = handler.action          # 当前动作（由 handler 提供）
    last_action = action.copy()      # 上一次动作的副本
    while not handler.done:          # 当 handler 未标记完成时持续运行
        if not handler.sync:
            # 如果 handler 当前不同步，跳过这次循环但不退出，控制频率仍然受限
            rate_limiter.sleep()
            continue

        # 更新动作历史：先保存上次动作，然后读取当前 handler 的动作
        last_action[:] = action
        action[:] = handler.action

        max_var_action.append(max(abs(action - last_action)))


        # 仅当动作有明显变化时才记录数据，避免重复帧
        if np.max(np.abs(action - last_action)) > 1e-6:
            # [修复] 在添加数据前进行预处理
            # 1. 将 agent_pos 和 actions 转换为 float32
            # 2. 将图像缩放到 (256, 256)
            data_dict['/observations/agent_pos'].append(observation['agent_pos'].astype(np.float32))
            for camera in env_cls.cameras:
                resized_image = cv2.resize(
                    observation['pixels'][camera], 
                    (256, 256), 
                    interpolation=cv2.INTER_AREA
                )
                data_dict[f'/observations/pixels/{camera}'].append(resized_image)
            data_dict['/actions'].append(action.astype(np.float32))
        else:
            # 如果动作没有变化，则恢复 last_action（保持动作不变）
            action[:] = last_action

        # 将动作送入环境并获取下一步观测（符合 gym 接口的 step 返回值）
        observation, reward, terminated, truncated, info = env.step(action)

        # 渲染环境并使用 OpenCV 显示每个摄像头的图像（从 RGB 转为 BGR 以便 OpenCV 显示）
        env.render()
        for camera in env.cameras:
            cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # 按照设定频率休眠，控制循环频率
        rate_limiter.sleep()

    # 处理结束，关闭窗口并释放资源
    cv2.destroyAllWindows()
    handler.close()
    env.close()
    
    print("max_var_action:", max(max_var_action))  
    print("data_dict lengths:", {k: len(v) for k, v in data_dict.items()})
    
    print("data_dict keys and shapes:")
    for k, v in data_dict.items():
        print(f"  {k}: {[np.array(item).shape for item in v]}") 

    return data_dict


def write_to_pickle(env_cls: Type[Env], data_dict: dict):
    """
    将 data_dict 保存为简单的 pickle 文件。
    """
    # 2. 定义输出目录，使用新的 "_pickle" 后缀
    output_dir = Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name + "_pickle")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否有数据，如果没有则不保存
    episode_length = len(data_dict['/actions'])
    if episode_length == 0:
        print("⚠️  警告：没有录制任何数据。跳过 Pickle 文件保存。")
        return

    # 确定新文件的索引
    index = len([f for f in output_dir.iterdir() if f.is_file()])
    pkl_path = output_dir / Path(f"episode_{index:06d}.pkl")

    # 在写入之前，将数据字典中的数据列表转换为Numpy数组，以优化存储
    for key, value in data_dict.items():
        data_dict[key] = np.array(value)

    # 使用二进制写模式 ('wb') 打开文件并用 pickle 保存字典
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"已将数据成功保存到 Pickle 文件: {pkl_path}")


def main():
    # 入口：解析参数、获取环境类、执行 teleoperation 并将结果写入文件
    args = parse_args()

    env_cls = EnvFactory.get_strategies(args.env_type)

    data_dict = teleoperate(env_cls, args.handler_type)

    # 3. 调用新的保存函数，替换掉 write_to_h5
    write_to_pickle(env_cls, data_dict)


if __name__ == '__main__':
    main()
