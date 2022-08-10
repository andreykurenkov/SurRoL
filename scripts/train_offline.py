import d3rlpy
from d3rlpy.algos import SAC
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics.scorer import *
import gym
from torch.optim import Adam
from surrol.gym import *
import argparse
from surrol.her_goal_env_wrapper import HERGoalEnvWrapper
import numpy as np
from d3rlpy.preprocessing import MultiplyRewardScaler
from d3rlpy.online.explorers import NormalNoise
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.models.encoders import VectorEncoderFactory
from sklearn.model_selection import train_test_split
from d3rlpy.models.optimizers import OptimizerFactory

def evaluate_on_environment_for_success(algo, env, n_trials = 20):
    success_count = 0
    episode_rewards = []
    for _ in range(n_trials):
        observation = env.reset()
        episode_reward = 0.0
        is_success = False

        while True:
            action = algo.predict([observation])[0]
            observation, reward, done, info = env.step(action)
            is_success = is_success or info['is_success']
            episode_reward+=reward

            if done:
                break
        if is_success:
            success_count+=1
        print('Reward: %f | Success: %d'%(episode_reward, int(is_success)))
        episode_rewards.append(episode_reward)
    return float(success_count)/n_trials, float(np.mean(episode_rewards))


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PegTransfer-v0")
parser.add_argument("--dataset_name", default="demos_5k_step_obs.h5")
parser.add_argument("--n_steps", type=int, default=20000)
parser.add_argument("--n_steps_per_epoch", type=int, default=1000)
parser.add_argument("--model_name", default="bcq_model.pl")
args = parser.parse_args()

done_if_success = False
#algo = d3rlpy.algos.CQL(scaler="standard",
#                        reward_scaler=reward_scaler,
#                        tau=0.001,
#                        use_gpu=True)

#algo = d3rlpy.algos.BCQ(scaler="standard",
#                        use_gpu=True,
#                        reward_scaler=reward_scaler,
#                        action_scaler='min_max',
#                        tau=0.001,
#                        actor_learning_rate=0.0001)

optim_factory = OptimizerFactory(Adam, eps=0.00001, weight_decay=0.0001)
#algo = d3rlpy.algos.BC(scaler="standard",
#                       optim_factory=optim_factory,
#                       policy_type="deterministic",
#                       reward_scaler="standard")
encoder_factory = VectorEncoderFactory([256,256,256],use_dense=True)
reward_scaler = MultiplyRewardScaler(0.1)
algo = d3rlpy.algos.TD3PlusBC(scaler="standard",
                              action_scaler="min_max",
                              reward_scaler=reward_scaler,
                              actor_optim_factory=optim_factory,
                              critic_optim_factory=optim_factory,
                              actor_encoder_factory=encoder_factory,
                              gamma=1.0,
                              alpha=0.001,
                              tau=0.001,
                              batch_size=256,
                              n_critics=2,
                              n_steps=1,
                              actor_learning_rate=0.001,
                              critic_learning_rate=0.0001,
                              update_actor_interval=1,
                              use_gpu=True)
#algo = d3rlpy.algos.PLASWithPerturbation(scaler="standard",
#                                         reward_scaler=reward_scaler,
#                                         warmup_steps=20000,
#                                         gamma=1.0)

dataset = MDPDataset.load("datasets/" + args.dataset_name)
train_episodes, test_episodes = train_test_split(dataset, test_size=0.05)
env = gym.make(args.task)
wrapped_env = HERGoalEnvWrapper(env,
                                done_if_success=done_if_success,
                                success_as_reward=False)
eval_env = gym.make(args.task)
wrapped_eval_env = HERGoalEnvWrapper(eval_env,
                                     done_if_success=done_if_success,
                                     success_as_reward=True)
exploration_noise = NormalNoise(0,0.005)

scorers = {'action_diff_score': continuous_action_diff_scorer,
           'td_error_score': td_error_scorer,
           'average_value_estimation_score': average_value_estimation_scorer,
           'initial_state_value_estimation_scorer': initial_state_value_estimation_scorer,
           'environment': evaluate_on_environment(wrapped_eval_env)
           }

online_buffer = ReplayBuffer(2000000,wrapped_env, list(dataset.episodes))

algo.fit(train_episodes,
         eval_episodes=test_episodes,
         n_steps=args.n_steps,
         n_steps_per_epoch=args.n_steps_per_epoch,
         tensorboard_dir='logs/tb_logs',
         logdir='logs/d3rlpy',
         scorers = scorers)

mean_success, mean_reward = evaluate_on_environment_for_success(algo,
                                                                wrapped_env, 25)
print('Mean reward of offline trained policy:%f'%mean_reward)
print('Mean success of offline trained policy:%f'%mean_success)
algo.fit_online(wrapped_env,
                buffer=online_buffer*4,
                n_steps=args.n_steps*2,
                n_steps_per_epoch=args.n_steps_per_epoch,
                eval_env=wrapped_eval_env,
                tensorboard_dir='logs/tb_logs',
                logdir='logs/d3rlpy',
                explorer=exploration_noise)
mean_success, mean_reward = evaluate_on_environment_for_success(algo,
                                                                wrapped_env, 25)

print('Mean reward of online fine-tuned policy:%f'%mean_reward)
print('Mean success of online fine-tuned policy:%f'%mean_success)
algo.save_model("models/" + args.model_name)
