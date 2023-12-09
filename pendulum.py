import gym

env = gym.make('Pendulum-v1', render_mode='human')
observation, info = env.reset(seed=11)
for _ in range(200):
    env.render()
    print('Obervation: ', observation, ' Info: ', info)
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print('Reward:', reward, ' Terminated: ', terminated, ' Truncated: ', truncated)

    if terminated or truncated:
        observation, info = env.reset()

env.close()