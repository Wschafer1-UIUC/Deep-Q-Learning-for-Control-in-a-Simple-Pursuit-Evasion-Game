#######################################################################################
# Filename: simMDP.py
#
# Description: This script contains functions to run the Pursuit-Evasion environment.
#
# Functions:
#   - simEnv()
#
#######################################################################################
import numpy as np
import torch
import imageio

## Simulate Pursuit-Evasion Environment ##
def simEnv(env, policy, max_steps=1000, render=False):
    #################################################################################
    # Function: simEnv
    #
    # Description: This function simulates a single episode in an environment.
    #
    # Inputs:
    #   env:          environment instance (must implement reset() and step())
    #   policy:       (callable) policy for environment state space
    #   max_steps:    (int) maximum number of steps taken in an episode
    #
    # Outputs: states, actions, rewards, eps_return
    #
    #################################################################################

    # initialize the environment
    obs = env.reset()
    states = [obs]
    actions = []
    rewards = []
    eps_return = 0.0
    frames = []

    # initial render of the environment (if desired)
    if render: 
        env.render()
        frames.append(capture_frame(env))

    # run simulation for one episode
    for _ in range(max_steps):

        # get the action, next observation (state), and reward from the current observation
        a = policy(obs)
        next_obs, r, terminated, truncated = env.step(a)

        actions.append(a)
        rewards.append(float(r))
        eps_return += float(r)
        states.append(next_obs)

        # render the current frame of the environment (if desired)
        if render: 
            env.render()
            frames.append(capture_frame(env))

        # end the episode early if terminal state is reached or gymnasium environment breaks (truncated)
        done = terminated or truncated
        if done:
            break

        # step forward
        obs = next_obs

    env.close()

    # save rendering as GIF
    imageio.mimsave('test_run.gif', frames, fps=10)

    return states, actions, rewards, eps_return

# Function to Extract and Make a Greedy Policy ##
def make_greedy_policy(Q, device):
    def policy(obs):
        with torch.no_grad():
            s = np.array(obs, dtype=np.float32)
            s_torch = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = Q(s_torch)
            return int(q_values.argmax(dim=1).item())
    return policy

## Function to Capture Current Environment Frame ##
def capture_frame(env):
    if env.fig is None:
        env.render()
    fig = env.fig
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    return rgb

