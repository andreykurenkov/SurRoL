#create log dir
log_dir = '/scr-ssd/alina/leval_logs'
os.makedirs(log_dir, exist_ok=True)
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)

from stable_baselines3.common import results_plotter

# Helper from the library
results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "{args.alg}, {args.task}")
