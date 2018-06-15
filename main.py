#!/usr/bin/env python3

import os
import sys
import numpy as np
import gym
import retro
import params
from model import model
from worker import worker, replay_memory

def main():
    
    #ensure directory hierarchy is made
    dirs = ['logs', 'logs/model', 'logs/records', 'logs/tmp']
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)
        #clean directorys but keep hierarchy
        if not path == dirs[0] and not params.recover:
            for old_file in os.listdir(path):
                os.remove(os.path.join(path, old_file))

    #init the environment depending on the name
    if params.env_name in retro.list_games():
        #init env without recorder
        env = retro.make(game=params.env_name,
                state=params.env_state,
                use_restricted_actions=retro.ACTIONS_DISCRETE)
    else:
        env = gym.make(params.env_name)

    env.seed(params.seed)
    print('[+] environment %s initialized' % params.env_name)
    
    #since we are preprocessing state
    state = env.reset()
    n_actions = env.action_space.n #discrete env
    state_shape = worker(None).process_state(state).shape
    print('[*] state shape: %s --> %s\n[*] actions: %s' % (
            state.shape, state_shape, n_actions))
    agent = worker(model(state_shape, n_actions, recover=params.recover))
    print('[+] worker %s' % agent.model.status)

    agent.train(env, episodes=params.train_episodes, 
            max_steps=params.train_max_steps,
            batch_size=params.train_batch,
            print_interval=params.train_print_interval)

    #close train env, create new instance to record test games
    if params.env_name in retro.list_games():
        env.close()
        env = retro.make(game='SonicTheHedgehog-Genesis',
                state='GreenHillZone.Act1', 
                record=os.path.join(params.out_dir, 'records'),
                use_restricted_actions=retro.ACTIONS_DISCRETE)
    else:
        env.close()
        #func that indicates which episodes to record and write
        vc = lambda n: n in [int(x) for x in np.linspace(
                params.test_episodes-1, 0, params.test_records)] 

        #wrapper that records episodes
            #fails on reset if game not done
        env = gym.wrappers.Monitor(env, directory=os.path.join(
                params.out_dir, 'records'), force=True, video_callable=vc)

    #test and close the env
    agent.test(env, episodes=params.test_episodes, 
            max_steps=params.test_max_steps, out_dir=params.out_dir,
            print_interval=params.test_print_interval)
    env.close()

if __name__ == '__main__':
    main()

