import gym
from surrol.gym import *
import argparse
import d3rlpy
from stable_baselines3.common.noise import NormalActionNoise
from surrol.her_goal_env_wrapper import HERGoalEnvWrapper
from surrol.oracle_policy import OraclePolicy

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PegTransfer-v0")
parser.add_argument("--output_name", default="demos.h5")
parser.add_argument("--n_steps", type=int, default=5000)
args = parser.parse_args()

env = gym.make(args.task)
wrapped_env = HERGoalEnvWrapper(env)

# setup algorithm
action_noise = NormalActionNoise([0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.25, 0.25, 0.25, 0.05, 0.05])
oracle_policy = OraclePolicy(env, wrapped_env, action_noise)

# prepare experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=args.n_steps, env=wrapped_env)

# start data collection
oracle_policy.collect(wrapped_env,
                      buffer,
                      n_steps=args.n_steps)

# export as MDPDataset
dataset = buffer.to_mdp_dataset()

# save MDPDataset
dataset.dump("datasets/" + args.output_name)

