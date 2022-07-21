import argparse

parser = argparse.ArgumentParser()
parser.add_argument('task', required = True)
parser.add_argument('alg', required = True)

args = parser.parser.parse_arg()

if args.alg == 'DDPG':
    model = DDPG('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
    # Train the agent
elif args.alg == 'PPO':
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/")
elif args.alg == 'HER':
    model = DDPG('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),verbose=1, tensorboard_log="./logs/")

from stable_baselines3 import DDPG

from surrol.gym import * 
import gym 

env = gym.make('{args.task}') # create one process and corresponding env

eval_env = gym.make('{args.task}')  # create one process and corresponding env
model.learn(total_timesteps=int(2e5), 
    log_interval=1, 
    eval_env=eval_env, 
    eval_freq=1000, 
    n_eval_episodes=5, 
    tb_log_name='DDPG_{args.task}', 
    eval_log_path="eval_logs")

#create log dir
log_dir = '/scr-ssd/alina/leval_logs'
os.makedirs(log_dir, exist_ok=True)
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)

from stable_baselines3.common import results_plotter

# Helper from the library
results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "{args.alg}, {args.task}")

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results(log_dir)


