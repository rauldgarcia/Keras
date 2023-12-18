import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import statistics

env = gym.make('CartPole-v1')

ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

mutation_step_size = 0.02
population_size = 10
seed = 0
generations = 10

networks = []

for network in range(population_size):

    # Create the model
    model = Sequential()
    model.add(Dense(5, input_dim=ninputs, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(noutputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    rows_1 = len(model.layers[0].get_weights()[0])
    columns_1 = len(model.layers[0].get_weights()[0][0])
    rows_2 = len(model.layers[1].get_weights()[0])
    columns_2 = len(model.layers[1].get_weights()[0][0])
    rows_3 = len(model.layers[2].get_weights()[0])
    columns_3 = len(model.layers[2].get_weights()[0][0])

    # Evaluate the model
    observation, info = env.reset()
    rewards = []
    for episode in range(30):
        total_reward = 0

        for _ in range(200):
            
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
    
    networks.append([model, statistics.mean(rewards)])

# Sort the networks
networks.sort(key=lambda x:x[1], reverse=True) 

for g in range(generations):
    for l in range(int(population_size/2)):

        # Create the model
        model_new = Sequential()
        weights_1 = np.random.normal(0, mutation_step_size, (rows_1, columns_1))
        model_new.add(Dense(5, input_dim=ninputs, activation='relu', weights=[weights_1, np.ones(columns_1)]))
        weights_2 = np.random.normal(0, mutation_step_size, (rows_2, columns_2))
        model_new.add(Dense(5, activation='relu', weights=[weights_2, np.ones(columns_2)]))
        weights_3 = np.random.normal(0, mutation_step_size, (rows_3, columns_3))
        model_new.add(Dense(noutputs, activation='sigmoid', weights=[weights_3, np.ones(columns_3)]))
        model_new.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_new.summary()

        # Evaluate the model
        observation, info = env.reset()
        rewards = []
        for episode in range(30):
            total_reward = 0

            for _ in range(200):
                observation = np.reshape(observation, [1, len(observation)])
                action = model.predict(observation)

                if (isinstance(env.action_space, gym.spaces.box.Box)):
                    action = action
                else:
                    action = np.argmax(action)

                observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
                total_reward += reward

                if terminated or truncated:
                    observation, info = env.reset()
                    break
            
            rewards.append(total_reward)
        networks.append([model_new, statistics.mean(rewards)])

    networks.sort(key=lambda x:x[1], reverse=True)

    for l in range(int(population_size/2)):
        networks.pop()

print(networks)

model = networks[0][0]

# Evaluate the model
observation, info = env.reset()
rewards = []
for episode in range(30):
    total_reward = 0

    for _ in range(200):
        observation = np.reshape(observation, [1, len(observation)])
        action = model.predict(observation)

        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = action
        else:
            action = np.argmax(action)

        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            break
    
    rewards.append(total_reward)

print(statistics.mean(rewards))
plt.plot(rewards)
plt.show()