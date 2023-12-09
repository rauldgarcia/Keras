import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from collections import deque

class DQN:
    def __init__(self,
                 env,
                 hidden_size=16):
        
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.model = Sequential()
        self.model.add(Dense(hidden_size,
                             activation='relu',
                             input_dim=self.state_size))
        self.model.add(Dense(hidden_size,
                             activation='relu'))
        self.model.add(Dense(self.action_size,
                             activation='linear'))
        self.model.compile(loss='mse',
                           optimizer='Adam')
        self.model.summary()

    def __call__(self, s):

        s = np.reshape(s, [1, self.state_size])
        a = self.model.predict(s)
        a = np.reshape(a, [self.action_size])
        return a


class Memory:

    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Generate an array with a random choice of values from 0 to len(self.buffer)
        # Replace=False no replacement
        idx = np.random.choice(len(self.buffer),
                               size=batch_size,
                               replace=False)
        
        # Now take the respective experiences according to idx and put them in an array
        return [self.buffer[ii] for ii in idx]


def create_experience(replaybuffer, eps=0.1, max_episodes=1000):
    average_return = 0

    for _ in range(max_episodes):
        done = False
        state, _ = env.reset()

        while not done:
            average_return += 1
            if eps > np.random.rand():
                action = env.action_space.sample()
            else:
                action = np.argmax(q(state))
            
            next_state, reward, done, terminate, info = env.step(action)
            replaybuffer.add((state, action, reward, next_state, done))
            state = next_state
    
    avg_return = average_return / max_episodes

    return avg_return


def eps_decay(i,
              start=1.0,
              stop=0.01,
              annealig_stop=1000):
    inew = min(i, annealig_stop)
    return (start * (annealig_stop - inew) + stop * inew) / annealig_stop

env = gym.make('CartPole-v0')
q = DQN(env, hidden_size=16)

train_episodes = 30 # Max number of episodes to learn from
gamma = 0.99 # Future reward discount

# Exploration parameters
explore_start = 0.9 # Exploration probability at start
explore_stop = 0.01 # Minimum exploration prob
annealing_stop = train_episodes/2

# Network parameters
hidden_size = 16 # Number of units in each Q-network hidden

# Memory parameters
batch_size = 32 # Experience mini_batch size

replaybuffer = Memory()
create_experience(replaybuffer, eps=1.0, max_episodes=1000)
learning_curve = [] # This is top collect the total rewards while training

for ep in range(1, train_episodes+1):
    done = False
    state, _ = env.reset()
    eps = eps_decay(ep, explore_start, explore_stop, annealing_stop)
    total_return = 0 # Calculate return in each episode

    while not done:
        #average_return += 1
        if eps > np.random.rand():
            action = env.action_space.sample()
        else:
            action = np.argmax(q(state))
        
        next_state, reward, done, terminate, info = env.step(action)
        total_return += reward
        replaybuffer.add((state, action, reward, next_state, done))
        state = next_state

        inputs = np.zeros((batch_size, env.observation_space.shape[0]))
        targets = np.zeros((batch_size, env.action_space.n))
        minibatch = replaybuffer.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b, done_b) in enumerate(minibatch):
            inputs[i:i+1] = state_b
            if done_b:
                target = reward_b
            else:
                target = reward_b + gamma * np.argmax(q(next_state_b)) # Bellman equation

            targets[i] = q(state_b)
            targets[i][action_b] = target

        q.model.fit(inputs, targets, epochs=1, verbose=0)
        
    if ep%10 == 0:
        print('Episode', ep, 'avg return', total_return)
    learning_curve.append(total_return)

plt.plot(learning_curve)
plt.show()