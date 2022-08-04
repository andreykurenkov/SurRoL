import d3rlpy
from d3rlpy.algos import SAC
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment, continuous_action_diff_scorer
import gym
from surrol.gym import *
import argparse
from surrol.her_goal_env_wrapper import HERGoalEnvWrapper
import numpy as np
from d3rlpy.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PegTransfer-v0")
parser.add_argument("--dataset_name", default="demos.h5")
parser.add_argument("--n_steps", type=int, default=10000)
parser.add_argument("--n_steps_per_epoch", type=int, default=1000)
parser.add_argument("--model_name", default="bcq_model.pl")
args = parser.parse_args()

dataset = MDPDataset.load("datasets/" + args.dataset_name)
mean = np.mean(dataset.observations, axis=0, keepdims=True)
std = np.std(dataset.observations, axis=0, keepdims=True)
scaler = StandardScaler(mean=mean, std=std)

# setup algorithm
algo = d3rlpy.algos.BCQ(scaler=scaler,
                        reward_scaler="standard")


# train
algo.fit(dataset,
         eval_episodes=dataset.episodes,
         n_steps=args.n_steps,
         n_steps_per_epoch=args.n_steps_per_epoch,
         tensorboard_dir='runs',
         scorers = {'action_diff': continuous_action_diff_scorer})
algo.save_model("models/" + args.model_name)

env = gym.make(args.task)
wrapped_env = HERGoalEnvWrapper(env, scaler)
env_eval = evaluate_on_environment(wrapped_env, render=True)
mean_reward = env_eval(algo)
print(mean_reward)
