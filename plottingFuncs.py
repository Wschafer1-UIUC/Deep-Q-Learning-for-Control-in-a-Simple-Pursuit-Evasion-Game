#######################################################################################
# Filename: plottingFuncs.py
#
# Description: This script contains functions for plotting metrics for RL algorithms.
#
#######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import re
import copy
from simMDP import *
import torch
import torch.nn as nn
import imageio


## Function to Encode the Current State as a Flattened Array ##
def encode_state(s):
    return np.array(s, dtype=np.float32)

## Function to Build the Neural Network ##
def build_neural_network(nS, nA, hidden_layer_sizes=(64,64,64)):

    # initialize network layer info
    layers = []
    current_layer_size = nS

    # add hidden layers in order
    for next_layer_size in hidden_layer_sizes:
        layers.append(nn.Linear(current_layer_size, next_layer_size))
        layers.append(nn.ReLU())
        current_layer_size = next_layer_size
    
    # add the final layer (states -> actions)
    layers.append(nn.Linear(current_layer_size, nA))

    # turn layers into pytorch neural net
    NN = nn.Sequential(*layers) 

    return NN

## Plot the Evaluation Return Vs. Time Steps ##
def plotEvalReturn(pi_sets, pi_names, env_ctor, hidden_layer_sizes, num_policies=10, gamma=0.95, eval_episodes=50, max_steps=1000, device=None, env_name="PursuitEvasionEnv", show_plot=False):

    # device selection
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # helper function to compute the discounted return
    def episode_return(gamma, rewards):
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    # get basic env info from a temporary instance
    env = env_ctor()
    obs_space = env.observation_space
    act_space = env.action_space
    nS = obs_space.shape[0]
    nA = act_space.n
    env.close()

    # plot the evaluation return vs. time steps
    fig = plt.figure()
    for alg_pis, name in zip(pi_sets, pi_names):

        # choose which policies to evaluate
        num_available = len(alg_pis)
        if num_policies >= num_available:
            idxs = list(range(num_available))
        else:
            idxs = np.linspace(0, num_available - 1, num_policies).astype(int).tolist()

        x_vals = []
        y_vals = []

        for idx in idxs:
            steps, policy_repr = alg_pis[idx]

            # build the policy network
            if isinstance(policy_repr, dict):
                pi = build_neural_network(nS, nA, hidden_layer_sizes).to(device)
                pi.load_state_dict(policy_repr)
            else:
                pi = copy.deepcopy(policy_repr).to(device)

            pi.eval()

            # evaluate this policy by greedy actions
            env = env_ctor()
            ep_returns = []

            for _ in range(eval_episodes):
                obs = env.reset()
                rewards = []

                for _ in range(max_steps):
                    s = encode_state(obs)
                    s_torch = torch.tensor(
                        s, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    with torch.no_grad():
                        logits = pi(s_torch)
                        action = int(logits.argmax(1).item())

                    obs, r, terminated, truncated = env.step(action)
                    rewards.append(float(r))

                    if terminated or truncated:
                        break

                ep_returns.append(episode_return(gamma, rewards))

            env.close()
            x_vals.append(int(steps))
            y_vals.append(float(np.mean(ep_returns)))

        plt.plot(x_vals, y_vals, label=name, linewidth=2, marker='D', markersize=5)

    plt.xlabel("Time Steps")
    plt.ylabel(f"Evaluation Return (mean over {eval_episodes} episodes)")
    plt.title(f"[{env_name}] Evaluation Return vs. Time Steps {pi_names}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.xlim(left=0)
    if show_plot:
        plt.show()

    # save the figure
    picName = f"{env_name} Evaluation Return vs Steps {pi_names}"
    safe_name = re.sub(r"[^\w\-]+", "_", picName).strip("_") + ".png"
    fig.savefig(safe_name, dpi=200)

    return safe_name

## Render a GIF of Stitched Policy Runs ##
def renderPoliciesGIF(pi, env_ctor, device, num_episodes=5, max_steps=500, gif_path="policy_rollouts.gif", fps=10, slowdown_factor=3):

    # helper function
    def capture_frame_from_fig(fig):
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)
        rgb = buf[:, :, 1:4]
        return rgb

    # initialize values
    pi.eval()
    all_frames = []

    for ep in range(num_episodes):
        env = env_ctor()
        obs = env.reset()

        # initial render
        env.render()
        if hasattr(env, "ax") and hasattr(env, "fig"):
            env.ax.text(
                0.02, 0.95,
                f"Run {ep+1}",
                transform=env.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        frame = capture_frame_from_fig(env.fig)
        for _ in range(slowdown_factor):
            all_frames.append(frame)

        for _ in range(max_steps):
            with torch.no_grad():
                s = np.array(obs, dtype=np.float32)
                s_torch = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = pi(s_torch)
                action = int(q_values.argmax(dim=1).item())

            obs, _, terminated, truncated = env.step(action)

            env.render()
            if hasattr(env, "ax") and hasattr(env, "fig"):
                env.ax.text(
                    0.02, 0.95,
                    f"Run {ep+1}",
                    transform=env.ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
            frame = capture_frame_from_fig(env.fig)
            for _ in range(slowdown_factor):
                all_frames.append(frame)

            if terminated or truncated:
                break

        env.close()

    # save all frames as one long GIF
    imageio.mimsave(gif_path, all_frames, fps=fps)
    print(f"Saved rollouts GIF to: {gif_path}")


