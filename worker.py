
import sys, os
sys.dont_write_bytecode = True #remove before release
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore startup messages

import pandas as pd
import numpy as np

import misc

class worker():
    def __init__(self, model):
        self.model = model

    def train(self, env, episodes=100000, max_steps=100, 
            train_interval=20, print_interval=1000):
        misc.debug('training for %s episodes (%s steps max)' 
                % (episodes, max_steps))
        batch = replay_memory(env.action_space.n)
        all_stats = []
        for episode in range(episodes):
            done = False
            state = batch.process_state(env.reset()) #batch does all others
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
            while not done and step < max_steps:
                #do action
                action, _ = self.model.action(state)
                next_state, reward, done, _ = env.step(action)
                reward = 0 if done else reward

                #FIXME: q-learning reward discount
                    #reward = reward + discount * Q(next state)
                _, value = self.model.action(batch.process_state(
                        next_state)) #FIXME: double processes next_state
                reward += 0.9 * value

                #add experience to batch
                batch.add((state, action, reward, done, next_state))

                if batch.size == train_interval:
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
                
            #learn on remaining steps
            #if batch.size > 0:
            #    loss = self.model.learn(batch.get())
            all_stats.append(stats)
        
        #output training stats
        #for stat in all_stats:
            #stat = pd.DataFrame(data=stat)
            #print(stat.describe().loc[['min', 'max', 'mean', 'std']])

    def test(self, env, episodes=100, max_steps=100):
        misc.debug('testing for %s episodes (%s steps max)' 
                % (episodes, max_steps))
        #init a dict of useful measurements
        stats = {
            'step': [],
            'reward': [],
        }
        batch = replay_memory(env.action_space.n)
        for episode in range(episodes):
            done = False
            state = batch.process_state(env.reset())
            reward_sum = 0
            step = 0
            while not done and step < max_steps:
                #do action
                action, _ = self.model.action(batch.process_state(state), 
                        epsilon=0.0)
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
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.size = 0

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.size = 0

    def add(self, experience):
        state, action_int, reward, done, next_state = experience
        
        #preprocess actions --> onhot, states --> nxn mat
        action = [0 for _ in range(self.n_actions)]
        action[action_int] = 0
        state = self.process_state(state)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.size += 1

    #FIXME: implement
    def get(self, discount=0.9):
        #propogate and decay rewards backward through time
            #if sparse, rewards = np.linspace(last, 0)
            #if no rewards in whole batch
                #no propogation?
            #reward for a state is function of next x rewards
                #and next x values predicted for states

        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        next_states = np.asarray(self.next_states, dtype=np.float32)
        return states, actions, rewards, dones, next_states

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

