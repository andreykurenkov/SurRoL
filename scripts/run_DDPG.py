import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', required = True)
args = parser.parser.parse_arg()

from stable_baselines3 import DDPG

from surrol.gym import * 
import gym 

env = gym.make('{args.name}') # create one process and corresponding env


model = DDPG('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
# Train the agent

eval_env = gym.make('{args.name}')  # create one process and corresponding env
model.learn(total_timesteps=int(2e5), 
    log_interval=1, 
    eval_env=eval_env, 
    eval_freq=1000, 
    n_eval_episodes=5, 
    tb_log_name='DDPG_{args.name}', 
    eval_log_path="eval_logs")
