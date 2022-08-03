import d3rlpy
from d3rlpy.algos import SAC
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment
import gym
from surrol.gym import *
import argparse
from surrol.her_goal_env_wrapper import HERGoalEnvWrapper

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument('dataset_name')
parser.add_argument('n_steps', type=int, default=10000)
parser.add_argument("model_name")
args = parser.parse_args()

dataset = MDPDataset.load("datasets/" + args.dataset_name)

# setup algorithm
sac = d3rlpy.algos.SAC(scaler="standard")

# train
sac.fit(dataset, n_steps=args.n_steps, tensorboard_dir='runs')
sac.save_model("models/" + args.model_name)

env = gym.make(args.task)
wrapped_env = HERGoalEnvWrapper(env)

# evaluate trained algorithm
mean_episode_return = evaluate_on_environment(wrapped_env, render=True)(sac)

print(mean_episode_return)