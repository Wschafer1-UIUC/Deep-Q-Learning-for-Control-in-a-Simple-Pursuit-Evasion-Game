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
\begin{figure}[H]
    \centering

    \begin{subfigure}[t]{0.49\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figs/EvalReturn_DQN_pursuer_homing_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png}
        \caption{Pursuer vs. Homing Evader}
        \label{fig:eval_homing}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.49\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figs/EvalReturn_DQN_pursuer_random_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png}
        \caption{Pursuer vs. Random Evader}
        \label{fig:eval_random}
    \end{subfigure}

    \vspace{0.5em}

    \begin{subfigure}[t]{0.49\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figs/EvalReturn_DQN_pursuer_alpha-blend_20000_0.1_[10.0, 10.0]_[10.0, 10.0]_[40.0, 40.0].png}
        \caption{Pursuer vs. Alpha-Blend Evader}
        \label{fig:eval_alpha}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.49\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figs/EvalReturn_DQN_pursuer_alpha-blend_30000_0.1_[5.0, 20.0]_[5.0, 20.0]_[40.0, 40.0].png}
        \caption{Pursuer vs. Alpha-Blend Evader (Random Velocities)}
        \label{fig:eval_alpha_rand}
    \end{subfigure}

    \caption{Evaluation return over training for DQN pursuers trained against different evader strategies.}
    \label{fig:training_performance}
\end{figure}

## Performance Results
