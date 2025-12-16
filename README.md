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

---

## Training Results
<p align="center">
  <img src="Evaluation_Returns/EvalReturn_DQN_pursuer_homing_20000_0.1_%5B10.0%2C%2010.0%5D_%5B10.0%2C%2010.0%5D_%5B40.0%2C%2040.0%5D.png" width="48%">
  <img src="Evaluation_Returns/EvalReturn_DQN_pursuer_random_20000_0.1_%5B10.0%2C%2010.0%5D_%5B10.0%2C%2010.0%5D_%5B40.0%2C%2040.0%5D.png" width="48%">
</p>

<p align="center">
  <img src="Evaluation_Returns/EvalReturn_DQN_pursuer_alpha-blend_20000_0.1_%5B10.0%2C%2010.0%5D_%5B10.0%2C%2010.0%5D_%5B40.0%2C%2040.0%5D.png" width="48%">
  <img src="Evaluation_Returns/EvalReturn_DQN_pursuer_alpha-blend_30000_0.1_%5B5.0%2C%2020.0%5D_%5B5.0%2C%2020.0%5D_%5B40.0%2C%2040.0%5D.png" width="48%">
</p>

<p align="center">
  <em>Figure: Evaluation return over training for DQN pursuers trained against different evader strategies.</em>
</p>

---

## Performance Results

### Quantitative
<p align="center">
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Homing_Pursuit_bar_success_rate.png" width="95%">
  <br>
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Homing_Pursuit_bar_steps_per_episode.png" width="95%">
  <br>
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Homing_Pursuit_bar_energy_per_episode.png" width="95%">
</p>

<p align="center">
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Deviated_Pursuit_bar_success_rate.png" width="95%">
  <br>
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Deviated_Pursuit_bar_steps_per_episode.png" width="95%">
  <br>
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Deviated_Pursuit_bar_energy_per_episode.png" width="95%">
</p>

<p align="center">
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Constant-Bearing_Pursuit_bar_success_rate.png" width="95%">
  <br>
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Constant-Bearing_Pursuit_bar_steps_per_episode.png" width="95%">
  <br>
  <img src="Algorithm_Comparisons/Charts/ALL_Analytical_Evasion_vs._Learned__Constant-Bearing_Pursuit_bar_energy_per_episode.png" width="95%">
</p>
