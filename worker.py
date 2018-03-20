
import sys, os
sys.dont_write_bytecode = True #remove before release
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore startup messages

import time
import pandas as pd
import numpy as np
import gym
import scipy.signal

import misc

class worker():
    def __init__(self, model):
        self.model = model

    def process_state(self, state):
        #convert to readable input (n x n matrix)
        dims = len(state.shape)
        if dims == 3: #rgb input --> greyscale
            r, g, b = state[:, :, 0], state[:, :, 1], state[:, :, 2]
            state = 0.2989 * r + 0.5870 * g + 0.1140 * b
            w = max(state.shape[0], state.shape[1])
            state.resize((w,w))
        elif dims == 2:
            w = max(state.shape[0], state.shape[1])
            state.resize((w,w))
        elif dims == 1:
            w = int(state.shape[0] / 2)
            state.resize((w, w))
        else:
            misc.fatal_error('state size unsupported: %s' % state.shape)
        return state

    def to_onehot(self, action, n_actions):
        oh = [0 for _ in range(n_actions)]
        oh[action] = 1
        return oh

    def train(self, env, episodes=10000, max_steps=10000, 
            train_interval=20, print_interval=1000):
        misc.debug('training for %s episodes (%s steps max)' 
                % (episodes, max_steps))
        start_time = time.time()
        batch = replay_memory()
        n_actions = env.action_space.n
        all_stats = []
        for episode in range(episodes):
            done = False
            state = self.process_state(env.reset())
            step = 0
            reward_sum = 0
            #init a dict of useful measurements
            stats = {
                'step': [],
                'reward': [],
                'loss': [],
            }
            if episode % print_interval == 0:
                misc.debug('episode %s' % episode)
                #self.test(env, episodes=10, max_steps=10000, records=0)
            while not done and step < max_steps:
                #do action
                action, value = self.model.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = 0 if done else reward

                #process observation data
                next_state = self.process_state(next_state)
                action = self.to_onehot(action, n_actions)

                #add experience to batch
                batch.add((state, action, reward, value, done, next_state))

                if batch.size == train_interval or done:
                    loss = self.model.learn(batch.get())
                    stats['loss'].append(loss)
                    batch.clear()

                #update
                step += 1
                next_state = state
                reward_sum += reward

            #episode stats
            stats['step'].append(step)
            stats['reward'].append(reward_sum)
                
            all_stats.append(stats)
        
        end_time = time.time()
        train_time = end_time - start_time
        train_mins = train_time / 60
        train_secs = train_time % 60
        misc.debug('finished training in %0.3sm %0.3ss (%0.3ss)' % (
                train_mins, train_secs, train_time))
        #FIXME: output training stats
        #for stat in all_stats:
            #stat = pd.DataFrame(data=stat)
            #print(stat.describe().loc[['min', 'max', 'mean', 'std']])

    def test(self, env, episodes=100, max_steps=10000, records=4, 
            out_dir='./logs'):
        misc.debug('testing for %s episodes (%s steps max)' 
                % (episodes, max_steps))
        #func that indicates which episodes to record and write
        vc = lambda n: n in [int(x) for x in np.linspace(episodes-1, 0, 
                records)] 
        #wrapper that records episodes
        env = gym.wrappers.Monitor(env, directory=out_dir, force=True, 
                video_callable=vc)
        #init a dict of useful measurements
        stats = {
            'step': [],
            'reward': [],
        }
        for episode in range(episodes):
            done = False
            state = self.process_state(env.reset())
            reward_sum = 0
            step = 0
            #wrapper fails on reset if game goes past max step
                #gym imposes internal max step anyways
            while not done: #and step < max_steps:
                #do action
                action, _ = self.model.act(self.process_state(state))
                state, reward, done, _ = env.step(action)
                
                #update
                reward_sum += reward
                step += 1
                
            #record episode stats
            stats['step'].append(step)
            stats['reward'].append(reward_sum)

        #ez output format
        stats = pd.DataFrame(data=stats)
        print(stats.describe().loc[['min', 'max', 'mean', 'std']])

class replay_memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_states = []
        self.size = 0

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_states = []
        self.size = 0

    def add(self, experience):
        state, action, reward, value, done, next_state = experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.size += 1

    def get(self):
        #calc advantage and store in the values array
        reward = 0.0
        if not self.dones[-1]:
            reward = self.values[-1]
        for i in range(self.size - 1, -1, -1): #reverse iterate
            reward = self.rewards[i] + 0.99 * reward
            self.rewards[i] = reward
            self.values[i] = reward - self.values[i]

        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        advantages = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        next_states = np.asarray(self.next_states, dtype=np.float32)
        return states, actions, rewards, advantages, dones, next_states

