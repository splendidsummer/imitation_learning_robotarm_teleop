from imitation_learning_lerobot.envs import DishwasherEnv

if __name__ == '__main__':
    env = DishwasherEnv(render_mode="human")
    env.run()
