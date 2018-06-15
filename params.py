
#environment list for ez switching
envs = ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 
        'BipedalWalker-v2', 'Pong-v4', 'SpaceInvaders-v0', 
        'Breakout-v0']
#retro.list_games()
#retro.list_states(game)

env_name = envs[0]#'SonicTheHedgehog-Genesis'
env_state = 'GreenHillZone.Act1' #only valid for retro environments

#training
train_episodes = 1000
train_print_interval = 100
train_max_steps = 500
save_interval = 10 #episode interval to save the model

#testing
test_episodes = 100
test_print_interval = 10
test_max_steps = 500
test_records = 4 #number of episodes to dump to video, disabled for retro

#hyper
train_batch = 20 #replay memory batch size
reward_decay = 0.99
learn_rate = 1e-4
#reward_offset = -1 #scalar added to raw environment rewards
#done_reward = 1000 #scalar for reaching done state

#misc
seed = 42
out_dir = './logs' #base folder for model, any recordings, etc
downsample = 'slow' #slow, fast, none. 'fast' sacrifices quality for speed
recover = False


