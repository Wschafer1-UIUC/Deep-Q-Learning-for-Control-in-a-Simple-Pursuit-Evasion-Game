#######################################################################################
# Filename: TRAINING.py
#
# Description: This script contains the code used for training and visualizing the DQN
#              algo on the Pursuit-Evasion environment.
#
#######################################################################################
import numpy as np
from pursuit_evasion_env import PursuitEvasionEnv
from DQN_algos import *
from simMDP import *
from plottingFuncs import *

## <================================= USER INPUTS =================================> ##

# Environment Configs
config = {
    "controlled_agent": "pursuer",
    "dt": 0.1,
    "max_steps": 500,
    "x_e_range": 0.0,
    "y_e_range": 0.0,
    "evader_dist": [40.0, 40.0],
    "command_center_radius": 3.0,
    "evader_radius": 1.0,
    "w_p": np.pi,
    "w_e": np.pi / 2.0,
    "k_pursuer": 1.0,
    "k_evader": 1.0,
    "evader_algo": "random",
    "pursuer_algo": "constant-bearing",
    "v_p": [20.0, 20.0],
    "v_e": [10.0, 10.0]
}

# DQN Training Params
num_episodes=20000
gamma=0.9
epsilon=0.0
epsilon_final=0.0
epsilon_decay_steps=10000
alpha=1e-4
hidden_layer_sizes=(64, 64)
max_buffer_size=50000
batch_size=64
q_target_update_freq=5000
replay_start_size=1000
model_save_name = f"DQN_{config['controlled_agent']}_{config["evader_algo"]}_{num_episodes}_{config["dt"]}_{config['v_p']}_{config["v_e"]}_{config["evader_dist"]}"    # DQN_ControlledAgent_EvaderAlgo_NumberOfEpisodes_TimeStep_VP_VE_EvaderDist
save_path = f"Trained_Models\\{model_save_name}"

# Evaluation Return Params
num_policies=50
eval_episodes=20
env_name="Pursuit-Evasion DQN Larger"

# Render Policy Params
num_render_episodes=10
gif_path=f"PolicyRollouts_{model_save_name}.gif"
fps=10
slowdown_factor=1

## <===============================================================================> ##



## <============================ TRAINING & RENDERING =============================> ##

# DQN Training Function
Q_online, snapshots = DQN(
    num_episodes=num_episodes,
    max_steps=config["max_steps"],
    env=PursuitEvasionEnv(config=config),
    gamma=gamma,
    epsilon=epsilon,
    epsilon_final=epsilon_final,
    epsilon_decay_steps=epsilon_decay_steps,
    alpha=alpha,
    hidden_layer_sizes=hidden_layer_sizes,
    max_buffer_size=max_buffer_size,
    batch_size=batch_size,
    q_target_update_freq=q_target_update_freq,
    replay_start_size=replay_start_size,
)
torch.save({"model_state_dict": Q_online.state_dict(), "config": config, "hidden_layer_sizes": hidden_layer_sizes}, f'{save_path}.pt')

# Plot Evaluation Return vs. Env Steps
plotEvalReturn(
    pi_sets=[snapshots],
    pi_names=["DQN"],
    env_ctor=lambda: PursuitEvasionEnv(config=config),
    hidden_layer_sizes=hidden_layer_sizes,
    num_policies=num_policies,
    gamma=gamma,
    eval_episodes=eval_episodes,
    max_steps=config["max_steps"],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    env_name=env_name,
    show_plot=False,
    save_name=f'EvalReturn_{model_save_name}.png'
)

# Render the Final Policy
renderPoliciesGIF(
    pi=Q_online, 
    env_ctor=lambda: PursuitEvasionEnv(config=config), 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    num_episodes=num_render_episodes, 
    max_steps=config["max_steps"], 
    gif_path=gif_path, 
    fps=fps,
    slowdown_factor=slowdown_factor
    )

## <===============================================================================> ##

