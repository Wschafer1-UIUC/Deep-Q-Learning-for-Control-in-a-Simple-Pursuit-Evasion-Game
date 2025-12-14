#######################################################################################
# Filename: TRAINING.py
#
# Description: This script contains the code used for training and visualizing the DQN
#              algo on the Pursuit-Evasion environment.
#
#######################################################################################
import numpy as np
from DQN_algos import *
from plottingFuncs import *
from plottingFuncs import compare_learned_vs_analytic

## <================================= USER INPUTS =================================> ##

# NOTE: 
#       (1) Test every evader + trained pursuer algorithm.   [alpha-blend, alpha-blend w/ random velocities, random, homing]
#       (2) Test every analytical pursuer algorithm.         [constant-bearing, deviated, homing]

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
    "w_p": np.pi / 2.0,
    "w_e": np.pi / 2.0,
    "k_pursuer": 1.0,
    "k_evader": 1.0,
    "evader_algo": "random",         # alpha-blend, random, homing
    "pursuer_algo": "deviated",   # constant-bearing, deviated, homing
    "v_p": [10.0, 10.0],
    "v_e": [10.0, 10.0]
}
testing_model_path = "DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt"
num_test_episodes = 1000
num_render_episodes = 5

## <===============================================================================> ##



## <============================ TESTING & VALIDATION =============================> ##

learned_results, analytical_results = compare_learned_vs_analytic(
    checkpoint_path=f'Trained_Models\\{testing_model_path}',
    base_config=config,
    evader_algo=config["evader_algo"],
    num_test_episodes=num_test_episodes,
    num_render_episodes=num_render_episodes,
    max_steps=config["max_steps"],
    gif_path=f"Algorithm_Comparisons\\Side_by_Side\\Side_by_Side_{config['evader_algo']}_vs_{config['pursuer_algo']}_AND_{testing_model_path}.gif",
    traj_fig_path=f"Algorithm_Comparisons\\Trajectory\\Trajectory_{config['evader_algo']}_vs_{config['pursuer_algo']}_AND_{testing_model_path}.jpg",
    fps=10,
    slowdown_factor=1,
)

print('\n')
actions_L = []; steps_L = []; successes_L = []; w_p_L = []; w_e_L = []; steps_L_succ = []; w_p_L_succ = []
for ep_data in learned_results:
    a = ep_data['actions']
    s = ep_data['steps']
    S = ep_data['success']
    wp = ep_data['w_p_history']
    we = ep_data['w_e_history']
    successes_L.append(S)
    actions_L.append(a)
    steps_L.append(s)
    w_p_L.append(wp)
    w_e_L.append(we)
    if S == True:
        steps_L_succ.append(s)
        w_p_L_succ.append(wp)

actions_A = []; steps_A = []; successes_A = []; w_p_A = []; w_e_A = []; steps_A_succ = []; w_p_A_succ = []
for ep_data in analytical_results:
    a = ep_data['actions']
    s = ep_data['steps']
    S = ep_data['success']
    wp = ep_data['w_p_history']
    we = ep_data['w_e_history']
    successes_A.append(S)
    actions_A.append(a)
    steps_A.append(s)
    w_p_A.append(wp)
    w_e_A.append(we)
    if S == True:
        steps_A_succ.append(s)
        w_p_A_succ.append(wp)
    
# compute the average number of steps per episode
avg_steps_per_episode_L = np.mean(steps_L)
avg_steps_per_episode_A = np.mean(steps_A)
avg_steps_per_episode_L_succ = np.mean(steps_L_succ)
avg_steps_per_episode_A_succ = np.mean(steps_A_succ)
med_steps_per_episode_L = np.median(steps_L)
med_steps_per_episode_A = np.median(steps_A)
med_steps_per_episode_L_succ = np.median(steps_L_succ)
med_steps_per_episode_A_succ = np.median(steps_A_succ)
print(f'Steps per Episode (Learned): (avg: {avg_steps_per_episode_L}, median: {med_steps_per_episode_L}) {steps_L}')
print(f'Steps per Episode (Analytic): (avg: {avg_steps_per_episode_A}, median: {med_steps_per_episode_A}) {steps_A}')
print(f'Steps per Successful Episode (Learned): (avg: {avg_steps_per_episode_L_succ}, median: {med_steps_per_episode_L_succ}) {steps_L_succ}')
print(f'Steps per Successful Episode (Analytic): (avg: {avg_steps_per_episode_A_succ}, median: {med_steps_per_episode_A_succ}) {steps_A_succ}')

# compute the number of successes in total
num_success_L = 0
num_success_A = 0
for ep_L, ep_A in zip(successes_L, successes_A):
    if ep_L == True:
        num_success_L += 1
    if ep_A == True:
        num_success_A += 1
print(f'Success Rate (Learned): {num_success_L}/{len(successes_L)}')
print(f'Success Rate (Analytic): {num_success_A}/{len(successes_A)}')

# compute the total amount of energy (turn rate) consumed
w_p_L_per_ep = []; w_p_A_per_ep = []
for wpL, wpA in zip(w_p_L, w_p_A):
    w_p_L_per_ep.append(float(sum(wpL)))
    w_p_A_per_ep.append(float(sum(wpA)))
avg_energy_use_per_ep_L = np.mean(w_p_L_per_ep)
avg_energy_use_per_ep_A = np.mean(w_p_A_per_ep)
med_energy_use_per_ep_L = np.median(w_p_L_per_ep)
med_energy_use_per_ep_A = np.median(w_p_A_per_ep)
print(f'Energy Use (Learned): (avg: {avg_energy_use_per_ep_L:.2f}, median: {med_energy_use_per_ep_L:.2f}) {w_p_L_per_ep}')
print(f'Energy Use (Analytic): (avg: {avg_energy_use_per_ep_A:.2f}, median: {med_energy_use_per_ep_A:.2f}) {w_p_A_per_ep}')

w_p_L_per_ep_succ = []; w_p_A_per_ep_succ = []
for wpL_succ in w_p_L_succ:
    w_p_L_per_ep_succ.append(float(sum(wpL_succ)))
for wpA_succ in w_p_A_succ:
    w_p_A_per_ep_succ.append(float(sum(wpA_succ)))
avg_energy_use_per_ep_L_succ = np.mean(w_p_L_per_ep_succ)
avg_energy_use_per_ep_A_succ = np.mean(w_p_A_per_ep_succ)
med_energy_use_per_ep_L_succ = np.median(w_p_L_per_ep_succ)
med_energy_use_per_ep_A_succ = np.median(w_p_A_per_ep_succ)
print(f'Successful Energy Use (Learned): (avg: {avg_energy_use_per_ep_L_succ:.2f}, median: {med_energy_use_per_ep_L_succ:.2f}) {w_p_L_per_ep_succ}')
print(f'Successful Energy Use (Analytic): (avg: {avg_energy_use_per_ep_A_succ:.2f}, median: {med_energy_use_per_ep_A_succ:.2f}) {w_p_A_per_ep_succ}')

# save results to text file
summary_lines = [
    f'Steps per Episode (Learned): (avg: {avg_steps_per_episode_L}, median: {med_steps_per_episode_L}) {steps_L}',
    f'Steps per Episode (Analytic): (avg: {avg_steps_per_episode_A}, median: {med_steps_per_episode_A}) {steps_A}',
    "",
    f'Steps per Successful Episode (Learned): (avg: {avg_steps_per_episode_L_succ}, median: {med_steps_per_episode_L_succ}) {steps_L_succ}',
    f'Steps per Successful Episode (Analytic): (avg: {avg_steps_per_episode_A_succ}, median: {med_steps_per_episode_A_succ}) {steps_A_succ}',
    "",
    f'Energy Use (Learned): (avg: {avg_energy_use_per_ep_L:.2f}, median: {med_energy_use_per_ep_L:.2f}) {w_p_L_per_ep}',
    f'Energy Use (Analytic): (avg: {avg_energy_use_per_ep_A:.2f}, median: {med_energy_use_per_ep_A:.2f}) {w_p_A_per_ep}',
    "",
    f'Successful Energy Use (Learned): (avg: {avg_energy_use_per_ep_L_succ:.2f}, median: {med_energy_use_per_ep_L_succ:.2f}) {w_p_L_per_ep_succ}',
    f'Successful Energy Use (Analytic): (avg: {avg_energy_use_per_ep_A_succ:.2f}, median: {med_energy_use_per_ep_A_succ:.2f}) {w_p_A_per_ep_succ}',
    "",
    f"Success Rate (Learned):  {num_success_L}/{len(successes_L)}",
    f"Success Rate (Analytic): {num_success_A}/{len(successes_A)}"
]
summary_text = "\n".join(summary_lines)
results_txt_path = f"Algorithm_Comparisons\\Results\\Summary_{config['evader_algo']}_vs_{config['pursuer_algo']}_AND_{testing_model_path}.txt"
with open(results_txt_path, "w") as f: f.write(summary_text)
print(f"Saved comparison summary to: {results_txt_path}\n")

## <===============================================================================> ##