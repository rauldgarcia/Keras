import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import History
import statistics

env = gym.make('CartPole-v1')

ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

print(ninputs)
print(noutputs)

history = History()
model = Sequential()
model.add(Dense(5, input_dim=ninputs, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(noutputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

observation, info = env.reset()
rewards = []
for episode in range(30):
    print('Episode: ', episode)
    total_reward = 0

    for _ in range(500):
        
        #print('Observation: ', observation, ' info: ', info)
        observation = np.reshape(observation, [1, len(observation)])
        action = model.predict(observation)

        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = action
        else:
            action = np.argmax(action)

        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        #print('Reward: ', reward, ' Terminated: ', terminated, 'Trucated', truncated)
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            break
    
    rewards.append(total_reward)

print(statistics.mean(rewards))
env.close()
plt.plot(rewards)
plt.show()