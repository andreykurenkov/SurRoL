from stable_baselines3 import PPO

from surrol.gym import * 
import gym 

env = gym.make('NeedleReach-v0') # create one process and corresponding env


model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
# Train the agent

eval_env = gym.make('NeedleReach-v0')  # create one process and corresponding env
model.learn(total_timesteps=int(2e5), 
    log_interval=1, 
    eval_env=eval_env, 
    eval_freq=1000, 
    n_eval_episodes=5, 
    tb_log_name='PPO_needlereach', 
    eval_log_path="eval_logs")




