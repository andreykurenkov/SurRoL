import argparse
from stable_baselines3 import DDPG, PPO, HerReplayBuffer, DQN, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from surrol.gym import *
import gym
import imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('alg')
args = parser.parse_args()

env = DummyVecEnv([lambda: gym.make(args.task)])
env = VecNormalize(env,
                   norm_obs=True,#This is the part that really matters
                   norm_reward=False,
                   clip_obs=200,
                   clip_reward=50.0,#no reward clipping
                   gamma=0.98,
                   epsilon=0.001)
action_noise = NormalActionNoise(0, 0.2)
if args.alg == 'DDPG':
    model = DDPG('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
    # Train the agent
elif args.alg == 'PPO':
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
elif args.alg == 'HER':
    model = DDPG('MultiInputPolicy',
        env,
        buffer_size=1000000,
        learning_rate=0.001,
        gradient_steps=40,
        train_freq=(2, 'episode'),
        tau=0.05,
        batch_size=256,
        gamma=0.98,
        action_noise=action_noise,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=50,
        ),
        verbose=1,
        tensorboard_log="./logs/",
        policy_kwargs={'net_arch': [256, 256, 256]})

eval_env = DummyVecEnv([lambda: gym.make(args.task)])
eval_env = VecNormalize(eval_env,
                   norm_obs=True,
                   norm_reward=False,
                   clip_obs=200,
                   clip_reward=50.0,
                   gamma=0.98,
                   epsilon=0.001)
model.learn(total_timesteps=int(10000),
    log_interval=10,
    eval_env=eval_env,
    eval_freq=1000,
    n_eval_episodes=10,
    tb_log_name='%s_%s'%(args.alg, args.task),
    eval_log_path="eval_logs/%s_%s"%(args.alg, args.task))

images = []
obs = env.reset()
img = env.render(mode='rgb_array')

for i in range(350):
    print('Step %d'%i)
    images.append(img)
    action, _ = model.predict(obs)
    if (i+1)%50==0:
        obs = env.reset()
    else:
        obs, _, _ ,_ = env.step(action)
    img = env.render(mode='rgb_array')

gif_path="eval_logs/%s_%s/run.gif"%(args.alg, args.task)

imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)


