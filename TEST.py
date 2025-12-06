import numpy as np
import time
from pursuit_evasion_env import PursuitEvasionEnv
from DQN_algos import *
from simMDP import *
from plottingFuncs import *

config = {
    "controlled_agent": "pursuer",
    "dt": 0.2,
    "max_steps": 500,
    "x_e_range": 0.0,
    "y_e_range": 0.0,
    "evader_dist": 20.0,
    "command_center_radius": 1.0,
    "evader_radius": 1.0,
    "w_p": np.pi / 2.0,
    "w_e": np.pi / 2.0,
    "k_pursuer": 1.0,
    "k_evader": 1.0,
    "v_p": [10.0, 10.0],
    "v_e": [10.0, 10.0]
}

Q_online, snapshots = DQN(
    num_episodes=10000,   # 30,000 episodes for good behavior
    max_steps=config["max_steps"],
    env=PursuitEvasionEnv(config=config),
    gamma=0.9,
    epsilon=0.0,
    epsilon_final=0.0,
    epsilon_decay_steps=10000,
    alpha=1e-4,
    hidden_layer_sizes=(64, 64),
    max_buffer_size=50000,
    batch_size=64,
    q_target_update_freq=5000,
    replay_start_size=1000,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_online = Q_online.to(device)
Q_online.eval()
env_ctor = lambda: PursuitEvasionEnv(config=config)
policy = make_greedy_policy(Q_online, device)
t_start = time.perf_counter()
states, actions, rewards, G = simEnv(
    PursuitEvasionEnv(config=config),
    policy,
    max_steps=config["max_steps"],
    render=True,
)
total_time = time.perf_counter() - t_start

print(f"Eval episode return: {G:.2f}")
print(f"Eval simulation time: {total_time:.5f} seconds.\n")

plotEvalReturn(
    pi_sets=[snapshots],
    pi_names=["DQN"],
    env_ctor=env_ctor,
    hidden_layer_sizes=(64, 64),
    num_policies=50,
    gamma=0.9,
    eval_episodes=20,
    max_steps=config["max_steps"],
    device=device,
    env_name="Pursuit-Evasion DQN",
    show_plot=False
)

renderPoliciesGIF(
    pi=Q_online, 
    env_ctor=env_ctor, 
    device=device, 
    num_episodes=5, 
    max_steps=500, 
    gif_path="policy_rollouts.gif", 
    fps=10,
    slowdown_factor=2
    )
