import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import History
import statistics
import random
import time
import copy

begin = time.time()

class Ann(Sequential):

    def __init__(self, child_weights=None):
        super().__init__()

        if child_weights is None:
            self.add(Dense(16, input_dim=ninputs, activation='relu'))
            self.add(Dense(16, activation='relu'))
            self.add(Dense(5, activation='sigmoid'))
        else:
            self.add(Dense(16, input_dim=ninputs, activation='relu', weights=[child_weights[0], np.ones(16)]))
            self.add(Dense(16, activation='relu', weights=[child_weights[1], np.ones(16)]))
            self.add(Dense(5, activation='sigmoid', weights=[child_weights[2], np.ones(5)]))

def crossover(ann1, ann2):

    ann1_weights = []
    ann2_weights = []
    child_weights = []

    for layer in ann1.layers:
        ann1_weights.append(layer.get_weights()[0])

    for layer in ann2.layers:
        ann2_weights.append(layer.get_weights()[0])

    for i in range(len(ann1_weights)):
        split = random.randint(0, np.shape(ann1_weights[i])[1]-1)

        for j in range(split, np.shape(ann1_weights[i])[1]-2):
            ann1_weights[i][:, j] = copy.deepcopy(ann2_weights[i][:, j]) # Maybe it will be neccesary to use copy.deepcopy

        child_weights.append(ann1_weights[i]) # Here too, append a deep copy

    mutation(child_weights)

    return Ann(child_weights)

def mutation(child_weights):
    selection = random.randint(0, len(child_weights)-1)
    mut = random.uniform(0, 1)
    if mut <= 0.05:
        child_weights[selection] += random.randint(2, 5)
    else:
        pass

env = gym.make('CartPole-v1')

ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

print(ninputs)
print(noutputs)

networks = []
pool = []
generations = 10
population = 10

for ind in range(population):
    ann = Ann()
    networks.append(ann)
    pool.append([ann, 0])

for generation in range(generations):

    for model in networks:
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

        pool.append([model, statistics.mean(rewards)])

    networks.clear()
    pool.sort(key=lambda x:x[1], reverse=True)
    pool = pool[:population]

    # Top 5, ramdomly select 2 parents
    for top in range(5):
        for h in range(2):
            r = random.randint(0, population-1)
            temp = crossover(pool[top][0], pool[r][0])
            networks.append(temp)

print(pool)

model = pool[0][0]

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

print("The running time is: ")
end = time.time()
print(end - begin)

print(statistics.mean(rewards))
plt.plot(rewards)
plt.title("Rewards from the best model trained with Genetic Algorithms.")
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.show()