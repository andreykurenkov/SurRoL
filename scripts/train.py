import argparse

parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('alg')

args = parser.parser.parse_arg()

if args.alg == 'DDPG':
    model = DDPG('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
    # Train the agent
elif args.alg == 'PPO':
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
elif args.alg == 'HER':
    model = DDPG('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),verbose=1, tensorboard_log="./logs/")

from stable_baselines3 import DDPG, PPO, HerReplayBuffer, DQN, SAC, TD3

from surrol.gym import * 
import gym 

env = gym.make('{args.task}') # create one process and corresponding env

eval_env = gym.make('{args.task}')  # create one process and corresponding env
model.learn(total_timesteps=int(2e5), 
    log_interval=1, 
    eval_env=eval_env, 
    eval_freq=1000, 
    n_eval_episodes=5, 
    tb_log_name='{args.alg}_{args.task}', 
    eval_log_path="eval_logs")
