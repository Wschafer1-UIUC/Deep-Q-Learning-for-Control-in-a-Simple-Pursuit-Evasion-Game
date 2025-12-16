# Evaluating Learned Pursuit Policies Against Classical Guidance Laws in a 1v1 Pursuit-Evasion Game

## Abstract
This study compares learning-based and analytical pursuit guidance in a 1v1 planar pursuit–
evasion game. A Deep Q-Learning pursuer is evaluated against classical homing, deviated, and
constant-bearing pursuit laws across deterministic and stochastic evasion strategies. Analytical
methods dominate structured scenarios, while learning-based pursuit improves robustness
under uncertainty at the cost of efficiency.

## Guidance Strategies
### Evader Guidance Strategies
- Homing Evasion
- Random Evasion
- Alpha-Blend Evasion

### Analytical Pursuit Guidace Laws
- Homing Pursuit
- Deviated Pursuit
- Constant-Bearing Pursuit

### Learned Guidance via Deep Q-Learning
- DQN-learned pursuit

## Training Results
<p align="center">
  <img src="figs/EvalReturn_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Pursuer vs. Homing Evader" width="48%">
  <img src="figs/EvalReturn_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Pursuer vs. Random Evader" width="48%">
</p>

<p align="center">
  <img src="figs/EvalReturn_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Pursuer vs. Alpha-Blend Evader" width="48%">
  <img src="figs/EvalReturn_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].png" alt="Pursuer vs. Alpha-Blend Evader (Random Velocities)" width="48%">
</p>

<p align="center">
  <em>Figure: Evaluation return over training for DQN pursuers trained against different evader strategies.</em>
</p>

## Performance Results

### Quantitative
<p align="center">
  <img src="figs/homing/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_success_rate.png" alt="Homing evasion: success rate" width="95%">
  <br>
  <img src="figs/homing/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_steps_per_episode.png" alt="Homing evasion: steps per episode" width="95%">
  <br>
  <img src="figs/homing/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_energy_per_episode.png" alt="Homing evasion: energy per episode" width="95%">
</p>
<p align="center">
  <img src="figs/random/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_success_rate.png" alt="Random evasion: success rate" width="95%">
  <br>
  <img src="figs/random/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_steps_per_episode.png" alt="Random evasion: steps per episode" width="95%">
  <br>
  <img src="figs/random/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_energy_per_episode.png" alt="Random evasion: energy per episode" width="95%">
</p>
<p align="center">
  <img src="figs/alpha-blend/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_success_rate.png" alt="Alpha-blend evasion: success rate" width="95%">
  <br>
  <img src="figs/alpha-blend/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_steps_per_episode.png" alt="Alpha-blend evasion: steps per episode" width="95%">
  <br>
  <img src="figs/alpha-blend/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_energy_per_episode.png" alt="Alpha-blend evasion: energy per episode" width="95%">
</p>
<p align="center">
  <img src="figs/alpha-blend_randV_[5.0,20.0]/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_success_rate.png" alt="Alpha-blend random velocities: success rate" width="95%">
  <br>
  <img src="figs/alpha-blend_randV_[5.0,20.0]/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_steps_per_episode.png" alt="Alpha-blend random velocities: steps per episode" width="95%">
  <br>
  <img src="figs/alpha-blend_randV_[5.0,20.0]/ALL_Analytical_Evasion_vs_Learned__Homing_Pursuit_bar_energy_per_episode.png" alt="Alpha-blend random velocities: energy per episode" width="95%">
</p>

### Qualitative
Side-by-side analytical vs. learned comparisons:
<p align="center">
  <img src="gifs/Side_by_Side_homing_vs_homing_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Homing evasion: homing pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_homing_vs_deviated_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Homing evasion: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_homing_vs_constant-bearing_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Homing evasion: constant-bearing pursuit vs DQN" width="95%">
</p>
<p align="center">
  <img src="gifs/Side_by_Side_random_vs_homing_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Random evasion: homing pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_random_vs_deviated_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Random evasion: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_random_vs_constant-bearing_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Random evasion: constant-bearing pursuit vs DQN" width="95%">
</p>
<p align="center">
  <img src="gifs/Side_by_Side_alpha-blend_vs_homing_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Alpha-blend evasion: homing pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_alpha-blend_vs_deviated_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Alpha-blend evasion: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_alpha-blend_vs_constant-bearing_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].gif" alt="Alpha-blend evasion: constant-bearing pursuit vs DQN" width="95%">
</p>

<p align="center">
  <img src="gifs/Side_by_Side_alpha-blend_vs_homing_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].gif" alt="Alpha-blend random velocities: homing pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_alpha-blend_vs_deviated_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].gif" alt="Alpha-blend random velocities: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="gifs/Side_by_Side_alpha-blend_vs_constant-bearing_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].gif" alt="Alpha-blend random velocities: constant-bearing pursuit vs DQN" width="95%">
</p>

Trajectories:

<p align="center">
  <img src="figs/Trajectory_homing_vs_homing_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Homing evasion trajectory: homing pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_homing_vs_deviated_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Homing evasion trajectory: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_homing_vs_constant-bearing_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Homing evasion trajectory: constant-bearing pursuit vs DQN" width="95%">
</p>
<p align="center">
  <img src="figs/Trajectory_random_vs_homing_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Random evasion trajectory: homing pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_random_vs_deviated_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Random evasion trajectory: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_random_vs_constant-bearing_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Random evasion trajectory: constant-bearing pursuit vs DQN" width="95%">
</p>
<p align="center">
  <img src="figs/Trajectory_alpha-blend_vs_homing_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Alpha-blend evasion trajectory: homing pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_alpha-blend_vs_deviated_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Alpha-blend evasion trajectory: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_alpha-blend_vs_constant-bearing_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png" alt="Alpha-blend evasion trajectory: constant-bearing pursuit vs DQN" width="95%">
</p>
<p align="center">
  <img src="figs/Trajectory_alpha-blend_vs_homing_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].png" alt="Alpha-blend random velocities trajectory: homing pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_alpha-blend_vs_deviated_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].png" alt="Alpha-blend random velocities trajectory: deviated pursuit vs DQN" width="95%">
  <br>
  <img src="figs/Trajectory_alpha-blend_vs_constant-bearing_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].png" alt="Alpha-blend random velocities trajectory: constant-bearing pursuit vs DQN" width="95%">
</p>

