
import sys, os
sys.dont_write_bytecode = True #remove before release
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore startup messages

import time
import pandas as pd
import numpy as np
import gym
import matplotlib.pyplot as plt
import skimage.measure

import params
import misc

class worker():
    def __init__(self, model):
        self.model = model

    def render_state(self, state):
        f, a = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
        f.tight_layout()
        a.imshow(state, cmap='gray')
        a.set_axis_off()
        plt.show()
        plt.close(f)

    def process_state(self, state, pad_value=0.0):
        #convert to standard size input (n x n matrix)
        #FIXME: should calc w once at start, not every call

        dims = len(state.shape)
        if dims == 3: #rgb input --> greyscale
            r, g, b = state[:, :, 0], state[:, :, 1], state[:, :, 2]
            state = 0.2989 * r + 0.5870 * g + 0.1140 * b
            w = max(state.shape[0], state.shape[1])
            new_state = np.full((w,w), pad_value)
            new_state[:state.shape[0], :state.shape[1]] = state
        elif dims == 2:
            w = max(state.shape[0], state.shape[1])
            new_state = np.full((w,w), pad_value)
            new_state[:state.shape[0], :state.shape[1]] = state
        elif dims == 1:
            w = 2
            while w**2 < state.shape[0]:
                w += 1
            state = np.reshape(state, (-1, w))
            new_state = np.full((w,w), pad_value)
            new_state[:state.shape[0], :state.shape[1]] = state
        else:
            misc.fatal_error('unsupported state size: %s' % state.shape)

        #only downsample if img input
        if params.downsample == 'slow' and dims > 1:
            new_state = skimage.measure.block_reduce(new_state, (2,2),
                    func=np.mean)
        elif params.downsample == 'fast' and dims > 1:
            new_state = new_state[::2,::2]

        return new_state

    def to_onehot(self, action, n_actions):
        oh = [0 for _ in range(n_actions)]
        oh[action] = 1
        return oh

    def train(self, env, episodes=10000, max_steps=10000, 
            batch_size=20, print_interval=1000):
        misc.debug('training for %s episodes (%s steps max)' 
                % (episodes, max_steps))
        train_start = time.time()
        batch = replay_memory()
        n_actions = env.action_space.n
        all_stats = []
        for episode in range(episodes):
            episode_start = time.time()
            state = self.process_state(env.reset())
            step = reward_sum = done = 0
            #init a dict of useful measurements
            stats = {'step': [], 'reward': [], 'loss': [],}
            while not done and step < max_steps:
                #do action
                action, value = self.model.act(state)
                next_state, reward, done, _ = env.step(action)

                #encourage either achieving or avoiding end state
                    #achieve when given negative rewards
                    #avoid when given positive rewards
                #reward = 0 if done else reward

                #reward += params.reward_offset

                #process observation data
                next_state = self.process_state(next_state)
                action = self.to_onehot(action, n_actions)

                #add experience to batch
                batch.add((state, action, reward, value, done, next_state))

                #learn
                if batch.size == batch_size or done:
                    loss = self.model.learn(batch.get())
                    stats['loss'].append(loss)
                    batch.clear()

                #update
                step += 1
                state = next_state
                reward_sum += reward

            #episode stats
            stats['step'].append(step)
            stats['reward'].append(reward_sum)
            self.model.add_episode_stat(reward_sum) #for tensorboard
                
            all_stats.append(stats)

            #save model on specified interval
            if (episode+1) % params.save_interval == 0:
                self.model.save()

            if (episode+1) % print_interval == 0:
                episode_time = time.time() - episode_start
                eta = episode_time * ((episodes-1) - episode)
                misc.debug(('episode %7s: %5s steps %3s reward in %5.5ss '
                        + '(ETA: %.3sm %.3ss)') % (
                        episode+1, step, reward_sum, episode_time, 
                        int(eta/60), eta%60))
        
        train_time = time.time() - train_start
        train_mins = int(train_time / 60)
        train_secs = train_time % 60
        misc.debug('finished training in %0.3sm %0.3ss (%0.5ss)' % (
                train_mins, train_secs, train_time))
        #FIXME: output training stats
        #for stat in all_stats:
            #stat = pd.DataFrame(data=stat)
            #print(stat.describe().loc[['min', 'max', 'mean', 'std']])

    def test(self, env, episodes=100, max_steps=10000, 
            out_dir='./logs', print_interval=10):
        misc.debug('testing for %s episodes (%s steps max)' 
                % (episodes, max_steps))

        #init a dict of useful measurements
        stats = {'step': [], 'reward': [],}
        test_start = time.time()
        for episode in range(episodes):
            episode_start = time.time()
            state = env.reset()
            step = reward_sum = done = 0
            while not done and step < max_steps:
                #do action
                action, _ = self.model.act(self.process_state(state), 
                        explore=False)
                state, reward, done, _ = env.step(action)
                
                #update
                reward_sum += reward
                step += 1
                
            #record episode stats
            stats['step'].append(step)
            stats['reward'].append(reward_sum)

            if (episode+1) % print_interval == 0:
                episode_time = time.time() - episode_start
                eta = episode_time * (episodes - episode)
                misc.debug(('episode %7s: %5s steps %3s reward in %5.5ss '
                        + '(ETA: %.3sm %.3ss)') % (
                        episode+1, step, reward_sum, episode_time, 
                        int(eta/60), eta%60))
        #timing 
        test_time = time.time() - test_start
        test_mins = int(test_time / 60)
        test_secs = test_time % 60
        misc.debug('finished testing in %0.3sm %0.3ss (%0.5ss)' % (
                test_mins, test_secs, test_time))
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
            reward = self.rewards[i] + params.reward_decay * reward
            self.rewards[i] = reward
            self.values[i] = reward - self.values[i]

        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        advantages = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        next_states = np.asarray(self.next_states, dtype=np.float32)
        return states, actions, rewards, advantages, dones, next_states

