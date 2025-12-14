#######################################################################################
# Filename: EVALUATION.py
#
# Description: This script contains the code used for evaluating and plotting the
#              results of analytical vs. trained pursuers on the Pursuit-Evasion 
#              environment.
#
#######################################################################################
from plottingFuncs import *

## Learned vs. Homing Pursuit ##
homing_summaries = {
        'Homing Evasion': r'Algorithm_Comparisons\Results\Summary_homing_vs_homing_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Random Evasion': r'Algorithm_Comparisons\Results\Summary_random_vs_homing_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Alpha-Blend Evasion': r'Algorithm_Comparisons\Results\Summary_alpha-blend_vs_homing_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Alpha-Blend Evasion w Randomized Velocity': r'Algorithm_Comparisons\Results\Summary_alpha-blend_vs_homing_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].pt.txt'
    }
homing_title = 'Analytical Evasion vs. Learned & Homing Pursuit'
plot_all_results_learned_vs_analytic(summaries = homing_summaries, title = homing_title)
plot_successful_results_learned_vs_analytic(summaries = homing_summaries, title = homing_title)

## Learned vs. Deviated Pursuit ##
deviated_summaries = {
        'Homing Evasion': r'Algorithm_Comparisons\Results\Summary_homing_vs_deviated_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Random Evasion': r'Algorithm_Comparisons\Results\Summary_random_vs_deviated_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Alpha-Blend Evasion': r'Algorithm_Comparisons\Results\Summary_alpha-blend_vs_deviated_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Alpha-Blend Evasion w Randomized Velocity': r'Algorithm_Comparisons\Results\Summary_alpha-blend_vs_deviated_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].pt.txt'
    }
deviated_title = 'Analytical Evasion vs. Learned & Deviated Pursuit'
plot_all_results_learned_vs_analytic(summaries = deviated_summaries, title = deviated_title)
plot_successful_results_learned_vs_analytic(summaries = deviated_summaries, title = deviated_title)

## Learned vs. Constant-Bearing Pursuit ##
constant_bearing_summaries = {
        'Homing Evasion': r'Algorithm_Comparisons\Results\Summary_homing_vs_constant-bearing_AND_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Random Evasion': r'Algorithm_Comparisons\Results\Summary_random_vs_constant-bearing_AND_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Alpha-Blend Evasion': r'Algorithm_Comparisons\Results\Summary_alpha-blend_vs_constant-bearing_AND_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].pt.txt',
        'Alpha-Blend Evasion w Randomized Velocity': r'Algorithm_Comparisons\Results\Summary_alpha-blend_vs_constant-bearing_AND_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].pt.txt'
    }
constant_bearing_title = 'Analytical Evasion vs. Learned & Constant-Bearing Pursuit'
plot_all_results_learned_vs_analytic(summaries = constant_bearing_summaries, title = constant_bearing_title)
plot_successful_results_learned_vs_analytic(summaries = constant_bearing_summaries, title = constant_bearing_title)
