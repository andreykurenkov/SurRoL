from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from surrol.gym import *
import gym
import imageio
import numpy as np
# the saved model does not contain the replay buffer

env = DummyVecEnv([lambda: gym.make("NeedleReach-v0")])
env = VecNormalize(env,
                   norm_obs=True,#This is the part that really matters
                   norm_reward=False,
                   clip_obs=200,
                   clip_reward=50.0,#no reward clipping
                   gamma=0.98,
                   epsilon=0.001)
loaded_model = DDPG.load("eval_logs/HER_NeedleReach-v0/best_model", env)

images = []
obs = env.reset()
img = env.render(mode='rgb_array')

for i in range(1000):
    print('Warmup Step %d'%i)
    action, _ = loaded_model.predict(obs)
    obs, _, _ ,_ = env.step(action)
    if (i+1)%50==0:
        obs = env.reset()
    else:
        obs, _, _ ,_ = env.step(action)

obs = env.reset()
for i in range(350):
    print('Step %d'%i)
    images.append(img)
    action, _ = loaded_model.predict(obs)
    if (i+1)%50==0:
        obs = env.reset()
    else:
        obs, _, _ ,_ = env.step(action)
    img = env.render(mode='rgb_array')

imageio.mimsave('needlereach_HER.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
