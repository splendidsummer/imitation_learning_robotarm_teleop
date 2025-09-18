from imitation_learning_lerobot.envs import BartendEnv

if __name__ == '__main__':
    env = BartendEnv(render_mode="human")
    env.run()
