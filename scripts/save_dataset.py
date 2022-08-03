import gym
from surrol.gym import *
import argparse
import d3rlpy

from surrol.her_goal_env_wrapper import HERGoalEnvWrapper
from surrol.oracle_policy import OraclePolicy

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument('output_name')
parser.add_argument('n_steps', type=int, default=10000)
args = parser.parse_args()

env = gym.make(args.task)
wrapped_env = HERGoalEnvWrapper(env)
#print(env.observation_space.shape)

# setup algorithm
oracle_policy = OraclePolicy(env, wrapped_env)
#random_policy = d3rlpy.algos.RandomPolicy()

# prepare experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=args.n_steps, env=wrapped_env)

# start data collection
oracle_policy.collect(wrapped_env, buffer, n_steps=args.n_steps)
#random_policy.collect(wrapped_env, buffer, n_steps=100)

# export as MDPDataset
dataset = buffer.to_mdp_dataset()

# save MDPDataset
dataset.dump("datasets/" + args.output_name)

