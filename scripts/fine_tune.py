import d3rlpy
from d3rlpy.algos import SAC
from d3rlpy.datasets import MDPDataset
from surrol.her_goal_env_wrapper import HERGoalEnvWrapper
import gym
from surrol.gym import *
import argparse
import numpy as np
from surrol.oracle_policy import OraclePolicy

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument('dataset_name')
parser.add_argument('n_steps', type=int, default=10000)
parser.add_argument("model_name")
args = parser.parse_args()

# load pretrained policy
dataset = MDPDataset.load("datasets/" + args.dataset_name)

# setup manually

env = gym.make(args.task)
wrapped_env = HERGoalEnvWrapper(env)

# setup algorithm
sac = d3rlpy.algos.SAC(scaler=scaler)

sac.build_with_env(wrapped_env)

sac.load_model("models/" + args.model_name)

# setup experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=args.n_steps, env=wrapped_env)

# setup exploration strategy if necessary
explorer = d3rlpy.online.explorers.ConstantEpsilonGreedy(0.1)

# start finetuning
sac.fit_online(wrapped_env, buffer, explorer, n_steps=args.n_steps,tensorboard_dir='runs')
