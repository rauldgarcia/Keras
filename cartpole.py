import gym
import numpy as np

env = gym.make('CartPole-v1', render_mode='human')

pvariance = 0.1 # Variance of initial parameters
ppvariance = 0.02 # Variance of pertubations
nhiddens = 5 # Number of internal neurons

ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

w1 = np.random.randn(nhiddens, ninputs) * pvariance
w2 = np.random.randn(noutputs, nhiddens) * pvariance
b1 = np.zeros(shape=(nhiddens,1))
b2 = np.zeros(shape=(nhiddens,1))

observation, info = env.reset(seed=42)

for _ in range(200):
    env.render()
    print('Observation: ', observation, ' info: ', info)

    observation.resizq(ninputs, 1)
    z1 = np.dot(w1, observation) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)

    if (isinstance(env.action_space, gym.spaces.box.Box)):
        action = a2
    else:
        action = np.argmax(a2)

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print('Reward: ', reward, ' Terminated: ', terminated, 'Trucated', truncated)

    if terminated or truncated:
        observation, info = env.reset()

env.close()