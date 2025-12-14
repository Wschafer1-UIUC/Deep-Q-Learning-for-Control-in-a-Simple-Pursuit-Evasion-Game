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
import torch
import torch.nn as nn
import imageio
import random
from pursuit_evasion_env import *
import os


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
def plotEvalReturn(pi_sets, pi_names, env_ctor, hidden_layer_sizes, num_policies=10, gamma=0.95, eval_episodes=50, max_steps=1000, device=None, env_name="PursuitEvasionEnv", show_plot=False, save_name=None):
    print('Starting Evaluation Return plot rendering ...')

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
    if save_name is None:
        save_name = re.sub(r"[^\w\-]+", "_", picName).strip("_") + ".png"
    fig.savefig(save_name, dpi=200)
    plt.close(fig)
    print(f'Rendered Evaluation Return plot. \n')

    return save_name

## Render a GIF of Stitched Policy Runs ##
def renderPoliciesGIF(pi, env_ctor, device, num_episodes=5, max_steps=500, gif_path="policy_rollouts.gif", fps=10, slowdown_factor=3):
    print(f'Starting policy rollout renderings ...')

    def capture_frame_from_fig(fig):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()
        return rgb

    pi.eval()
    all_frames = []

    for ep in range(num_episodes):
        print(f'{ep+1} episode(s) rendered ...')
        env = env_ctor()
        obs = env.reset()

        env.render(show=False)
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

            env.render(show=False)
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

    imageio.mimsave(gif_path, all_frames, fps=fps)
    print(f"Saved rollouts GIF to: {gif_path}\n")

## Generate Results for the Learned vs. Analytical Pursuer ##
def compare_learned_vs_analytic(checkpoint_path, base_config, evader_algo="homing", num_test_episodes=5, num_render_episodes=5, max_steps=500, gif_path="learned_vs_analytic_side_by_side.gif", traj_fig_path="traj_comparison.png", fps=10, slowdown_factor=3, device=None):
    print(f"\nLoading model from: {checkpoint_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state_dict = ckpt["model_state_dict"]
    hidden_layer_sizes = ckpt.get("hidden_layer_sizes", (64, 64))

    temp_config = base_config.copy()
    temp_config["controlled_agent"] = "pursuer"
    temp_config["evader_algo"] = evader_algo
    temp_env = PursuitEvasionEnv(config=temp_config)
    nS = temp_env.observation_space.shape[0]
    nA = temp_env.action_space.n
    temp_env.close()

    pi = build_neural_network(nS, nA, hidden_layer_sizes).to(device)
    pi.load_state_dict(model_state_dict)
    pi.eval()

    def capture_frame_from_fig(fig):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()
        return rgb

    learned_config = base_config.copy()
    learned_config["controlled_agent"] = "pursuer"
    learned_config["evader_algo"] = evader_algo

    analytic_config = base_config.copy()
    analytic_config["controlled_agent"] = "none"
    analytic_config["evader_algo"] = evader_algo

    learned_traj_p = None
    learned_traj_e = None
    analytic_traj_p = None
    analytic_traj_e = None
    traj_command_center = None

    combined_frames = []
    learned_results = []
    analytic_results = []

    num_render_episodes = min(num_render_episodes, num_test_episodes)
    base_seed = random.randint(10000, 99999)
    print("Starting comparison rollouts ...")
    for ep in range(num_test_episodes):
        print(f"Running episode {ep+1}/{num_test_episodes} ...")
        render_this = ep < num_render_episodes
        ep_seed = base_seed + ep
        env_L = PursuitEvasionEnv(config=learned_config)
        obs_L = env_L.reset(seed=ep_seed)
        env_A = PursuitEvasionEnv(config=analytic_config)
        obs_A = env_A.reset(seed=ep_seed)
        
        ep_L_p = []
        ep_L_e = []
        actions_L = []
        last_evader_captured_L = False
        last_evader_reached_L = False

        ep_A_p = []
        ep_A_e = []
        actions_A = []
        last_evader_captured_A = False
        last_evader_reached_A = False

        if render_this:
            env_L.render(show=False)
            if hasattr(env_L, "ax") and hasattr(env_L, "fig"):
                env_L.ax.text(0.02, 0.95, f"Run {ep+1} (Learned)", transform=env_L.ax.transAxes, ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            frame_L = capture_frame_from_fig(env_L.fig)

            env_A.render(show=False)
            if hasattr(env_A, "ax") and hasattr(env_A, "fig"):
                env_A.ax.text(0.02, 0.95, f"Run {ep+1} (Analytic CB)", transform=env_A.ax.transAxes, ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            frame_A = capture_frame_from_fig(env_A.fig)

        done_L = False
        done_A = False
        for _ in range(max_steps):
            if not done_L:
                with torch.no_grad():
                    s = np.array(obs_L, dtype=np.float32)
                    s_torch = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = pi(s_torch)
                    action_L = int(q_values.argmax(dim=1).item())

                prev_e_L = env_L.evader_state[:2].copy()
                prev_p_L = env_L.pursuer_state[:2].copy()

                obs_L, _, terminated_L, truncated_L = env_L.step(action_L)
                _, _, _, _, _, evader_reached_L, evader_captured_L = env_L.check_termination(prev_e_L, prev_p_L)

                last_evader_captured_L = evader_captured_L
                last_evader_reached_L = evader_reached_L
                actions_L.append(action_L)

                x_p_L, y_p_L, _, _ = env_L.pursuer_state
                x_e_L, y_e_L, _, _ = env_L.evader_state
                ep_L_p.append((x_p_L, y_p_L))
                ep_L_e.append((x_e_L, y_e_L))

                if render_this:
                    env_L.render(show=False)
                    if hasattr(env_L, "ax") and hasattr(env_L, "fig"):
                        env_L.ax.text(0.02, 0.95, f"Run {ep+1} (Learned)", transform=env_L.ax.transAxes, ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
                    frame_L = capture_frame_from_fig(env_L.fig)

                if terminated_L or truncated_L:
                    done_L = True

            if not done_A:
                prev_e_A = env_A.evader_state[:2].copy()
                prev_p_A = env_A.pursuer_state[:2].copy()

                _, _, terminated_A, truncated_A = env_A.step(1)
                _, _, _, _, _, evader_reached_A, evader_captured_A = env_A.check_termination(prev_e_A, prev_p_A)

                last_evader_captured_A = evader_captured_A
                last_evader_reached_A = evader_reached_A
                actions_A.append(1)

                x_p_A, y_p_A, _, _ = env_A.pursuer_state
                x_e_A, y_e_A, _, _ = env_A.evader_state
                ep_A_p.append((x_p_A, y_p_A))
                ep_A_e.append((x_e_A, y_e_A))

                if render_this:
                    env_A.render(show=False)
                    if hasattr(env_A, "ax") and hasattr(env_A, "fig"):
                        env_A.ax.text(0.02, 0.95, f"Run {ep+1} (Analytic CB)", transform=env_A.ax.transAxes, ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
                    frame_A = capture_frame_from_fig(env_A.fig)

                if terminated_A or truncated_A:
                    done_A = True

            if render_this:
                h = max(frame_L.shape[0], frame_A.shape[0])
                if frame_L.shape[0] < h:
                    pad = h - frame_L.shape[0]
                    frame_L_pad = np.pad(frame_L, ((0, pad), (0, 0), (0, 0)), mode="edge")
                else:
                    frame_L_pad = frame_L

                if frame_A.shape[0] < h:
                    pad = h - frame_A.shape[0]
                    frame_A_pad = np.pad(frame_A, ((0, pad), (0, 0), (0, 0)), mode="edge")
                else:
                    frame_A_pad = frame_A

                combined = np.concatenate([frame_L_pad, frame_A_pad], axis=1)
                for _ in range(slowdown_factor):
                    combined_frames.append(combined)

            if done_L and done_A:
                break

        w_p_L, w_e_L = env_L.get_integrated_control_histories()
        w_p_A, w_e_A = env_A.get_integrated_control_histories()

        cmd_center = env_L.command_center.copy()
        success_L = bool(last_evader_captured_L and not last_evader_reached_L)
        learned_results.append({"actions": actions_L, "steps": len(actions_L), "success": success_L, "w_p_history": w_p_L, "w_e_history": w_e_L})

        success_A = bool(last_evader_captured_A and not last_evader_reached_A)
        analytic_results.append({"actions": actions_A, "steps": len(actions_A), "success": success_A, "w_p_history": w_p_A, "w_e_history": w_e_A})

        env_L.close()
        env_A.close()

        if ep == 0:
            learned_traj_p = np.array(ep_L_p)
            learned_traj_e = np.array(ep_L_e)
            analytic_traj_p = np.array(ep_A_p)
            analytic_traj_e = np.array(ep_A_e)
            traj_command_center = cmd_center

    if len(combined_frames) > 0:
        imageio.mimsave(gif_path, combined_frames, fps=fps)
        print(f"Saved side-by-side comparison GIF to: {gif_path}")
    else:
        print("No frames rendered; GIF was not created (num_render_episodes = 0).")

    if (learned_traj_p is not None and learned_traj_e is not None and analytic_traj_p is not None and analytic_traj_e is not None):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect("equal")

        if traj_command_center is not None:
            ax.scatter(
                traj_command_center[0],
                traj_command_center[1],
                marker="s",
                s=80,
                c="k",
                label="Command Center",
            )

        ax.plot(
            learned_traj_p[:, 0],
            learned_traj_p[:, 1],
            color="tab:blue",
            label="Pursuer (learned)",
        )
        ax.plot(
            learned_traj_e[:, 0],
            learned_traj_e[:, 1],
            color="lightblue",
            linestyle="--",
            label="Evader (learned run)",
        )

        ax.plot(
            analytic_traj_p[:, 0],
            analytic_traj_p[:, 1],
            color="tab:red",
            label="Pursuer (analytic CB)",
        )
        ax.plot(
            analytic_traj_e[:, 0],
            analytic_traj_e[:, 1],
            color="salmon",
            linestyle="--",
            label="Evader (analytic run)",
        )

        ax.scatter(
            learned_traj_p[0, 0],
            learned_traj_p[0, 1],
            marker="o",
            s=40,
            color="tab:blue",
            label="_nolegend_",
        )
        ax.scatter(
            learned_traj_p[-1, 0],
            learned_traj_p[-1, 1],
            marker="x",
            s=40,
            color="tab:blue",
            label="_nolegend_",
        )
        ax.scatter(
            learned_traj_e[0, 0],
            learned_traj_e[0, 1],
            marker="o",
            s=40,
            color="lightblue",
            label="_nolegend_",
        )
        ax.scatter(
            learned_traj_e[-1, 0],
            learned_traj_e[-1, 1],
            marker="x",
            s=40,
            color="lightblue",
            label="_nolegend_",
        )

        ax.scatter(
            analytic_traj_p[0, 0],
            analytic_traj_p[0, 1],
            marker="o",
            s=40,
            color="tab:red",
            label="_nolegend_",
        )
        ax.scatter(
            analytic_traj_p[-1, 0],
            analytic_traj_p[-1, 1],
            marker="x",
            s=40,
            color="tab:red",
            label="_nolegend_",
        )
        ax.scatter(
            analytic_traj_e[0, 0],
            analytic_traj_e[0, 1],
            marker="o",
            s=40,
            color="salmon",
            label="_nolegend_",
        )
        ax.scatter(
            analytic_traj_e[-1, 0],
            analytic_traj_e[-1, 1],
            marker="x",
            s=40,
            color="salmon",
            label="_nolegend_",
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Learned vs Analytic Pursuit: Trajectory Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(traj_fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved trajectory comparison figure to: {traj_fig_path}")

    return learned_results, analytic_results

## Evaluate All Results for the Learned vs. Analytical Pursuer ##
def plot_all_results_learned_vs_analytic(summaries, title, out_dir=r'Algorithm_Comparisons\Charts'):

    def _nice_label(k: str) -> str:
        return k.replace("_", " ").strip()

    def _extract_avg(line: str) -> float:
        m = re.search(r"\(avg:\s*([-+]?[\d.]+(?:e[-+]?\d+)?|nan)\b", line, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not parse avg from line:\n{line}")
        val = m.group(1).lower()
        return float("nan") if val == "nan" else float(val)

    def _extract_rate(line: str) -> float:
        m = re.search(r":\s*([0-9]+)\s*/\s*([0-9]+)", line)
        if not m:
            raise ValueError(f"Could not parse success rate from line:\n{line}")
        num, den = int(m.group(1)), int(m.group(2))
        return num / den if den != 0 else 0.0

    def parse_summary_txt(fp: str) -> dict:
        with open(fp, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        data = {}
        for ln in lines:
            if ln.startswith("Steps per Episode (Learned):"):
                data["steps_all_learned"] = _extract_avg(ln)
            elif ln.startswith("Steps per Episode (Analytic):"):
                data["steps_all_analytic"] = _extract_avg(ln)
            elif ln.startswith("Steps per Successful Episode (Learned):"):
                data["steps_succ_learned"] = _extract_avg(ln)
            elif ln.startswith("Steps per Successful Episode (Analytic):"):
                data["steps_succ_analytic"] = _extract_avg(ln)
            elif ln.startswith("Energy Use (Learned):"):
                data["energy_all_learned"] = _extract_avg(ln)
            elif ln.startswith("Energy Use (Analytic):"):
                data["energy_all_analytic"] = _extract_avg(ln)
            elif ln.startswith("Successful Energy Use (Learned):"):
                data["energy_succ_learned"] = _extract_avg(ln)
            elif ln.startswith("Successful Energy Use (Analytic):"):
                data["energy_succ_analytic"] = _extract_avg(ln)
            elif ln.startswith("Success Rate (Learned):"):
                data["success_learned"] = _extract_rate(ln)
            elif ln.startswith("Success Rate (Analytic):"):
                data["success_analytic"] = _extract_rate(ln)

        required = [
            "steps_all_learned", "steps_all_analytic",
            "energy_all_learned", "energy_all_analytic",
            "success_learned", "success_analytic"
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing keys {missing} while parsing {fp}")

        if "steps_succ_learned" not in data:
            data["steps_succ_learned"] = float("nan")
        if "steps_succ_analytic" not in data:
            data["steps_succ_analytic"] = float("nan")
        if "energy_succ_learned" not in data:
            data["energy_succ_learned"] = float("nan")
        if "energy_succ_analytic" not in data:
            data["energy_succ_analytic"] = float("nan")

        return data

    def _sanitize_filename(s: str) -> str:
        s = re.sub(r"[^\w\-\. ]+", "", s)
        s = s.strip().replace(" ", "_")
        return s[:160] if len(s) > 160 else s

    def _evader_label_from_key(k: str) -> str:
        kk = k.lower()

        if ("alpha-blend" in kk or "alpha_blend" in kk) and ("random" in kk) and ("vel" in kk):
            return "Alpha-Blend (Random Velocities)"

        for token, pretty in [
            ("alpha-blend", "Alpha-Blend"),
            ("alpha_blend", "Alpha-Blend"),
            ("long random", "Long-Random"),
            ("long_random", "Long-Random"),
            ("random", "Random"),
            ("homing", "Homing"),
        ]:
            if token in kk:
                return pretty

        return _nice_label(k)

    def _nan_to_num(v: np.ndarray) -> np.ndarray:
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def _annotate_bars(ax, bars, raw_vals, fmt="{:.2f}", na_text="N/A", ypad_frac=0.01):
        ymin, ymax = ax.get_ylim()
        rng = max(ymax - ymin, 1e-9)
        for b, rv in zip(bars, raw_vals):
            if rv is None or (isinstance(rv, float) and np.isnan(rv)):
                txt = na_text
            else:
                txt = fmt.format(float(rv))
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2.0,
                h + ypad_frac * rng,
                txt,
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False
            )

    os.makedirs(out_dir, exist_ok=True)

    keys = list(summaries.keys())
    labels = [_evader_label_from_key(k) for k in keys]
    parsed = [parse_summary_txt(summaries[k]) for k in keys]

    steps_all_L = np.array([d["steps_all_learned"] for d in parsed], dtype=float)
    steps_suc_L = np.array([d["steps_succ_learned"] for d in parsed], dtype=float)
    steps_all_A = np.array([d["steps_all_analytic"] for d in parsed], dtype=float)
    steps_suc_A = np.array([d["steps_succ_analytic"] for d in parsed], dtype=float)

    energy_all_L = np.array([d["energy_all_learned"] for d in parsed], dtype=float)
    energy_suc_L = np.array([d["energy_succ_learned"] for d in parsed], dtype=float)
    energy_all_A = np.array([d["energy_all_analytic"] for d in parsed], dtype=float)
    energy_suc_A = np.array([d["energy_succ_analytic"] for d in parsed], dtype=float)

    succ_L = np.array([d["success_learned"] for d in parsed], dtype=float)
    succ_A = np.array([d["success_analytic"] for d in parsed], dtype=float)

    steps_all_L_plot = _nan_to_num(steps_all_L)
    steps_suc_L_plot = _nan_to_num(steps_suc_L)
    steps_all_A_plot = _nan_to_num(steps_all_A)
    steps_suc_A_plot = _nan_to_num(steps_suc_A)

    energy_all_L_plot = _nan_to_num(energy_all_L)
    energy_suc_L_plot = _nan_to_num(energy_suc_L)
    energy_all_A_plot = _nan_to_num(energy_all_A)
    energy_suc_A_plot = _nan_to_num(energy_suc_A)

    x = np.arange(len(keys), dtype=float)
    w = 0.18
    gap = 0.08

    o_all_L  = -(1.5 * w + gap / 2)
    o_all_A  = -(0.5 * w + gap / 2)
    o_suc_L  = +(0.5 * w + gap / 2)
    o_suc_A  = +(1.5 * w + gap / 2)

    c_L_all  = "tab:blue"
    c_L_suc  = "lightblue"
    c_A_all  = "tab:red"
    c_A_suc  = "salmon"

    def _finish(ax, chart_name: str, xlabel: str):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title}\n{chart_name}", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x + o_all_L, steps_all_L_plot, width=w, label="All Episodes (Learned)", color=c_L_all)
    b2 = ax.bar(x + o_all_A, steps_all_A_plot, width=w, label="All Episodes (Analytic)", color=c_A_all)
    b3 = ax.bar(x + o_suc_L, steps_suc_L_plot, width=w, label="Successful Only (Learned)", color=c_L_suc)
    b4 = ax.bar(x + o_suc_A, steps_suc_A_plot, width=w, label="Successful Only (Analytic)", color=c_A_suc)
    ax.set_ylabel("Steps")
    _finish(ax, "Steps per Episode", "Evader Algorithm")
    ax.set_ylim(0, max(np.max(steps_all_L_plot), np.max(steps_all_A_plot), np.max(steps_suc_L_plot), np.max(steps_suc_A_plot), 1.0) * 1.20)
    _annotate_bars(ax, b1, steps_all_L, fmt="{:.1f}")
    _annotate_bars(ax, b2, steps_all_A, fmt="{:.1f}")
    _annotate_bars(ax, b3, steps_suc_L, fmt="{:.1f}")
    _annotate_bars(ax, b4, steps_suc_A, fmt="{:.1f}")
    steps_path = os.path.join(out_dir, f"ALL_{_sanitize_filename(title)}_bar_steps_per_episode.png")
    fig.tight_layout()
    fig.savefig(steps_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x + o_all_L, energy_all_L_plot, width=w, label="All Episodes (Learned)", color=c_L_all)
    b2 = ax.bar(x + o_all_A, energy_all_A_plot, width=w, label="All Episodes (Analytic)", color=c_A_all)
    b3 = ax.bar(x + o_suc_L, energy_suc_L_plot, width=w, label="Successful Only (Learned)", color=c_L_suc)
    b4 = ax.bar(x + o_suc_A, energy_suc_A_plot, width=w, label="Successful Only (Analytic)", color=c_A_suc)
    ax.set_ylabel(r"Energy Usage Proxy  $\sum |\Delta \theta|$")
    _finish(ax, "Energy per Episode", "Evader Algorithm")
    ax.set_ylim(0, max(np.max(energy_all_L_plot), np.max(energy_all_A_plot), np.max(energy_suc_L_plot), np.max(energy_suc_A_plot), 1e-6) * 1.25)
    _annotate_bars(ax, b1, energy_all_L, fmt="{:.2f}")
    _annotate_bars(ax, b2, energy_all_A, fmt="{:.2f}")
    _annotate_bars(ax, b3, energy_suc_L, fmt="{:.2f}")
    _annotate_bars(ax, b4, energy_suc_A, fmt="{:.2f}")
    energy_path = os.path.join(out_dir, f"ALL_{_sanitize_filename(title)}_bar_energy_per_episode.png")
    fig.tight_layout()
    fig.savefig(energy_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    w2 = 0.35
    b1 = ax.bar(x - w2/2, succ_L, width=w2, label="Learned", color=c_L_all)
    b2 = ax.bar(x + w2/2, succ_A, width=w2, label="Analytic", color=c_A_all)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.10)
    _finish(ax, "Success Rate", "Evader Algorithm")
    _annotate_bars(ax, b1, succ_L, fmt="{:.3f}")
    _annotate_bars(ax, b2, succ_A, fmt="{:.3f}")
    succ_path = os.path.join(out_dir, f"ALL_{_sanitize_filename(title)}_bar_success_rate.png")
    fig.tight_layout()
    fig.savefig(succ_path, dpi=300)
    plt.close(fig)

    return {"steps_plot": steps_path, "energy_plot": energy_path, "success_plot": succ_path}

## Evaluate Successful Results for the Learned vs. Analytical Pursuer ##
def plot_successful_results_learned_vs_analytic(summaries, title, out_dir=r'Algorithm_Comparisons\Charts'):

    def _nice_label(k: str) -> str:
        return k.replace("_", " ").strip()

    def _extract_avg(line: str) -> float:
        m = re.search(r"\(avg:\s*([-+]?[\d.]+(?:e[-+]?\d+)?|nan)\b", line, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not parse avg from line:\n{line}")
        val = m.group(1).lower()
        return float("nan") if val == "nan" else float(val)

    def _extract_rate_and_counts(line: str):
        m = re.search(r":\s*([0-9]+)\s*/\s*([0-9]+)", line)
        if not m:
            raise ValueError(f"Could not parse success rate from line:\n{line}")
        num, den = int(m.group(1)), int(m.group(2))
        rate = num / den if den != 0 else 0.0
        return rate, num, den

    def parse_summary_txt(fp: str) -> dict:
        with open(fp, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        data = {}
        for ln in lines:
            if ln.startswith("Steps per Successful Episode (Learned):"):
                data["steps_succ_learned"] = _extract_avg(ln)
            elif ln.startswith("Steps per Successful Episode (Analytic):"):
                data["steps_succ_analytic"] = _extract_avg(ln)
            elif ln.startswith("Successful Energy Use (Learned):"):
                data["energy_succ_learned"] = _extract_avg(ln)
            elif ln.startswith("Successful Energy Use (Analytic):"):
                data["energy_succ_analytic"] = _extract_avg(ln)
            elif ln.startswith("Success Rate (Learned):"):
                r, n, d = _extract_rate_and_counts(ln)
                data["success_learned"] = r
                data["success_learned_n"] = n
                data["success_den"] = d
            elif ln.startswith("Success Rate (Analytic):"):
                r, n, d = _extract_rate_and_counts(ln)
                data["success_analytic"] = r
                data["success_analytic_n"] = n
                data["success_den"] = d

        required = [
            "steps_succ_learned", "steps_succ_analytic",
            "energy_succ_learned", "energy_succ_analytic",
            "success_learned", "success_analytic",
            "success_learned_n", "success_analytic_n", "success_den"
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing keys {missing} while parsing {fp}")

        return data

    def _sanitize_filename(s: str) -> str:
        s = re.sub(r"[^\w\-\. ]+", "", s)
        s = s.strip().replace(" ", "_")
        return s[:160] if len(s) > 160 else s

    def _evader_label_from_key(k: str) -> str:
        kk = k.lower()

        if ("alpha-blend" in kk or "alpha_blend" in kk) and ("random" in kk) and ("vel" in kk):
            return "Alpha-Blend (Random Velocities)"

        for token, pretty in [
            ("alpha-blend", "Alpha-Blend"),
            ("alpha_blend", "Alpha-Blend"),
            ("long random", "Long-Random"),
            ("long_random", "Long-Random"),
            ("random", "Random"),
            ("homing", "Homing"),
        ]:
            if token in kk:
                return pretty

        return _nice_label(k)

    def _nan_to_num(v: np.ndarray) -> np.ndarray:
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def _annotate_bars(ax, bars, raw_vals, fmt="{:.2f}", na_text="N/A", ypad_frac=0.015):
        ymin, ymax = ax.get_ylim()
        rng = max(ymax - ymin, 1e-9)
        for b, rv in zip(bars, raw_vals):
            if rv is None or (isinstance(rv, float) and np.isnan(rv)):
                txt = na_text
            else:
                txt = fmt.format(float(rv))
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2.0,
                h + ypad_frac * rng,
                txt,
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False
            )

    os.makedirs(out_dir, exist_ok=True)

    keys = list(summaries.keys())
    labels = [_evader_label_from_key(k) for k in keys]
    parsed = [parse_summary_txt(summaries[k]) for k in keys]

    steps_suc_L = np.array([d["steps_succ_learned"] for d in parsed], dtype=float)
    steps_suc_A = np.array([d["steps_succ_analytic"] for d in parsed], dtype=float)

    energy_suc_L = np.array([d["energy_succ_learned"] for d in parsed], dtype=float)
    energy_suc_A = np.array([d["energy_succ_analytic"] for d in parsed], dtype=float)

    succ_L = np.array([d["success_learned"] for d in parsed], dtype=float)
    succ_A = np.array([d["success_analytic"] for d in parsed], dtype=float)

    succ_L_n = np.array([d["success_learned_n"] for d in parsed], dtype=float)
    succ_A_n = np.array([d["success_analytic_n"] for d in parsed], dtype=float)
    succ_den = int(parsed[0]["success_den"])

    steps_suc_L_plot = _nan_to_num(steps_suc_L)
    steps_suc_A_plot = _nan_to_num(steps_suc_A)

    energy_suc_L_plot = _nan_to_num(energy_suc_L)
    energy_suc_A_plot = _nan_to_num(energy_suc_A)

    x = np.arange(len(keys), dtype=float)
    w = 0.35
    gap = 0.10

    o_L = -(0.5 * w + gap / 2)
    o_A = +(0.5 * w + gap / 2)

    c_L = "lightblue"
    c_A = "salmon"

    def _finish(ax, chart_name: str, xlabel: str):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title}\n{chart_name}", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x + o_L, steps_suc_L_plot, width=w, label="Learned", color=c_L)
    b2 = ax.bar(x + o_A, steps_suc_A_plot, width=w, label="Analytic", color=c_A)
    ax.set_ylabel("Steps")
    _finish(ax, "Steps per Successful Episode", "Evader Algorithm")
    ymax_steps = max(np.nanmax(steps_suc_L_plot), np.nanmax(steps_suc_A_plot), 1.0) * 1.25
    ax.set_ylim(0.0, ymax_steps)
    _annotate_bars(ax, b1, steps_suc_L, fmt="{:.1f}")
    _annotate_bars(ax, b2, steps_suc_A, fmt="{:.1f}")
    steps_path = os.path.join(out_dir, f"{_sanitize_filename(title)}_bar_steps_success_only.png")
    fig.tight_layout()
    fig.savefig(steps_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x + o_L, energy_suc_L_plot, width=w, label="Learned", color=c_L)
    b2 = ax.bar(x + o_A, energy_suc_A_plot, width=w, label="Analytic", color=c_A)
    ax.set_ylabel(r"Energy Usage Proxy  $\sum |\Delta \theta|$")
    _finish(ax, "Energy per Successful Episode", "Evader Algorithm")
    ymax_energy = max(np.nanmax(energy_suc_L_plot), np.nanmax(energy_suc_A_plot), 1e-6) * 1.30
    ax.set_ylim(0.0, ymax_energy)
    _annotate_bars(ax, b1, energy_suc_L, fmt="{:.2f}")
    _annotate_bars(ax, b2, energy_suc_A, fmt="{:.2f}")
    energy_path = os.path.join(out_dir, f"{_sanitize_filename(title)}_bar_energy_success_only.png")
    fig.tight_layout()
    fig.savefig(energy_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    w2 = 0.35
    b1 = ax.bar(x - w2/2, succ_L, width=w2, label="Learned", color="tab:blue")
    b2 = ax.bar(x + w2/2, succ_A, width=w2, label="Analytic", color="tab:red")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.10)
    _finish(ax, "Success Rate", "Evader Algorithm")
    _annotate_bars(ax, b1, succ_L_n, fmt="{:.0f}")
    _annotate_bars(ax, b2, succ_A_n, fmt="{:.0f}")
    ax.text(
        0.99, 0.02,
        f"Counts out of {succ_den}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        alpha=0.85
    )
    succ_path = os.path.join(out_dir, f"{_sanitize_filename(title)}_bar_success_rate.png")
    fig.tight_layout()
    fig.savefig(succ_path, dpi=300)
    plt.close(fig)

    return {"steps_plot": steps_path, "energy_plot": energy_path, "success_plot": succ_path}

