from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from surrol.gym import * 
import gym 

env = gym.make('NeedleReach-v0') # create one process and corresponding env

goal_selection_strategy = 'future'

online_sampling = True
# Time limit for the episodes
max_episode_length = 50

model = DDPG('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),verbose=1, tensorboard_log="./logs/")
# Train the agent

eval_env = gym.make('NeedleReach-v0')  # create one process and corresponding env
model.learn(total_timesteps=int(2e5), 
    log_interval=1, 
    eval_env=eval_env, 
    eval_freq=1000, 
    n_eval_episodes=5, 
    tb_log_name='HER_needlereach', 
    eval_log_path="eval_logs")