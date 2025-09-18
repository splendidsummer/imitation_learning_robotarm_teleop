from imitation_learning_lerobot.envs import PickAndPlaceEnv

if __name__ == '__main__':
    env = PickAndPlaceEnv(render_mode="human")
    env.run()
