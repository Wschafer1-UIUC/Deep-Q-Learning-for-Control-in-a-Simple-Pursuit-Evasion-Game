#######################################################################################
# Filename: pursuit-evasion-env.py
#
# Description: This script contains the PursuitEvasionEnv class used to run the custom
#              pursuit-evasion game for reinforcement learning applications.
#
#######################################################################################
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt


## Class Instantiating the Pursuit-Evasion Game ##
class PursuitEvasionEnv:
    ###################################################################################
    # Class: PursuitEvasionEnv
    #
    # Description: This class runs a pursuit-evasion game wherein a pursuer defends a
    #              command center being targeted by an evader. The evader's goal is to
    #              reach the command center before the pursuer can capture it.
    #              
    #              Observation Space: [ x_self, y_self, theta_self, v_self, ...
    #                                   x_other, y_other, theta_other, v_other ]
    #              
    #              Action Space: [0: turn left, 1: go straight, 2: turn right]
    #
    #              Reward Structure: { +100.0   for success (captured or reached) }
    #
    #              The command center is initialized within some range around the
    #              origin, the evader is initialized within some radius about the
    #              command center, and the pursuer is initialized at the location of
    #              the command center. The evader initially faces the command center
    #              and the pursuer initially faces the evader.
    #
    # Inputs:
    #   config = {
    #       "controlled_agent":      "pursuer" or "evader",
    #       "dt":                    float,
    #       "max_steps":             int,
    #       "x_e_range":             float,
    #       "y_e_range":             float,
    #       "evader_dist":           [dist_e_min, dist_e_max],
    #       "command_center_radius": float,
    #       "evader_radius":         float,
    #       "w_p":                   float,
    #       "w_e":                   float,
    #       "k_pursuer":             float,
    #       "k_evader":              float,
    #       "v_p":                   [v_p_min, v_p_max],
    #       "v_e":                   [v_e_min, v_e_max],
    #       "evader_algo":           "alpha-blend" or "homing" or "random" or "long random"
    #       "pursuer_algo":          "constant-bearing" or "" or ""
    #       "seed":                  int
    #   }
    #
    # Outputs:
    #   reset() = obs
    #   step(action) = obs, reward, terminated, truncated
    #
    ###################################################################################

    ## environment initialization ##
    def __init__(self, config=None):
        self.config = config or {}
        self.observation_space = None
        self.action_space = None
        self.random = None
        self.controlled_agent = self.config.get("controlled_agent", "pursuer")

        # environment state variables
        self.evader_state = None
        self.pursuer_state = None
        self.command_center = None
        self.t = None
        self.step_count = None

        # rendering state
        self.fig = None
        self.ax = None

        # environment reward structure
        self.reward_success = 1.0
        self.reward_failure = 0.0
        self.reward_d_ep = 0.0
        self.reward_d_ec = 0.0
        self.reward_d_pc = 0.0
        self.prev_d_ep = 0.0
        self.prev_d_ec = 0.0
        self.prev_d_pc = 0.0
        
        # episode configs
        self.dt = float(self.config.get("dt", 0.1))
        self.max_steps = int(self.config.get("max_steps", 500))

        # spawn configs
        self.x_e_range = float(self.config.get("x_e_range", 100.0))           # command center x spawn range [-100, 100]
        self.y_e_range = float(self.config.get("y_e_range", 100.0))           # command center y spawn range [-100, 100]

        # termination configs
        self.command_center_radius = float(self.config.get("command_center_radius", 3.0))   # size of the command center
        self.evader_radius = float(self.config.get("evader_radius", 1.0))                   # size of the evader

        # agent turn rate dynamics configs
        self.w_p_max = float(self.config.get("w_p", np.pi/4))   # rad/s
        self.w_e_max = float(self.config.get("w_e", np.pi/4))   # rad/s

        # agent guidance proportional control configs (default / non-learned guidance)
        self.k_pursuer = float(self.config.get("k_pursuer", 1.0))
        self.k_evader = float(self.config.get("k_evader", 1.0))

        # evader method (if not the controlled agent)
        self.evader_algo = self.config.get("evader_algo", "homing")   # homing, alpha-blend, random, long random
        self.dash_flag = False   
        self.curr_step = 0
        self.rand_steps = 0
        self.w_e_hold = 0.0

        # pursuer method (if not the controlled agent)
        self.pursuer_algo = self.config.get("pursuer_algo", "constant-bearing")   # constant-bearing, deviated, homing
        self.T_horizon = 0.75   # second

        # control logging
        self.w_p_hist = []
        self.w_e_hist = []

        # initialize agent observation and action space
        self._init_spaces()
        self._init_random_generator()

    ## observation and action space initialization ##
    def _init_spaces(self):
        
        # set agent velocity bounds
        self.v_p = self.config.get("v_p", [1.0, 5.0])
        self.v_e = self.config.get("v_e", [1.0, 5.0])
        self.v_p_min, self.v_p_max = float(self.v_p[0]), float(self.v_p[1])   # m/s
        self.v_e_min, self.v_e_max = float(self.v_e[0]), float(self.v_e[1])   # m/s

        # set the agent heading bounds
        theta_min = -np.pi  # radians
        theta_max = np.pi   # radians

        # set the agent position bounds
        x_min, x_max = -np.inf, np.inf   # m
        y_min, y_max = -np.inf, np.inf   # m

        # define the observation space
        if self.controlled_agent == "pursuer":
            v_min, v_max = self.v_p_min, self.v_p_max
        else:
            v_min, v_max = self.v_e_min, self.v_e_max
        low = np.array([x_min, y_min, theta_min, v_min, x_min, y_min, theta_min, v_min], dtype=np.float32)
        high = np.array([x_max, y_max, theta_max, v_max, x_max, y_max, theta_max, v_max], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(len(high),), dtype=np.float32)

        # define the discrete action space
        self.action_space = spaces.Discrete(3)   # (left, straight, right)

    ## seeded RNG initialization ##
    def _init_random_generator(self):
        seed = self.config.get("seed", None)
        self.random = np.random.default_rng(seed)

    ## environment reset ##
    def reset(self, seed=None, options=None):
        if seed is not None: self.random = np.random.default_rng(seed)

        # reset the time and environment steps
        self.t = 0.0
        self.step_count = 0

        # reset command center location
        x_c = self.random.uniform(-self.x_e_range, self.x_e_range)
        y_c = self.random.uniform(-self.y_e_range, self.y_e_range)
        self.command_center = np.array([x_c, y_c], dtype=np.float32)

        # reset evader location
        evader_dist_range = self.config.get("evader_dist", [10, 50])
        evader_dist_min, evader_dist_max = float(evader_dist_range[0]), float(evader_dist_range[1])
        evader_dist = self.random.uniform(evader_dist_min, evader_dist_max)

        # reset pursuer and evader forward velocities
        v_p = self.config.get("v_p", [1.0, 5.0])
        v_e = self.config.get("v_e", [1.0, 5.0])
        v_p_min, v_p_max = float(v_p[0]), float(v_p[1])   # m/s
        v_e_min, v_e_max = float(v_e[0]), float(v_e[1])   # m/s
        v_e = self.random.uniform(v_e_min, v_e_max)
        v_p = self.random.uniform(v_p_min, v_p_max)

        # reset evader position and heading
        rand_angle = self.random.uniform(0, 2*np.pi)
        x_e = x_c + evader_dist * np.cos(rand_angle)
        y_e = y_c + evader_dist * np.sin(rand_angle)
        theta_e = np.arctan2(y_c - y_e, x_c - x_e)

        # reset the pursuer position and heading
        x_p = x_c
        y_p = y_c
        theta_p = np.arctan2(y_e - y_p, x_e - x_p)

        # define pursuer and evader states
        self.evader_state = np.array([x_e, y_e, theta_e, v_e])
        self.pursuer_state = np.array([x_p, y_p, theta_p, v_p])

        # initialize previous distance for reward structure
        x_e, y_e, _, _ = self.evader_state
        x_p, y_p, _, _ = self.pursuer_state
        x_c, y_c = self.command_center

        d_ec = np.hypot(x_e - x_c, y_e - y_c)  # evader to command center
        d_ep = np.hypot(x_e - x_p, y_e - y_p)  # evader to pursuer
        d_pc = np.hypot(x_p - x_c, y_p - y_c)  # pursuer to command center

        if self.controlled_agent == "pursuer":
            self.prev_d_ep = d_ep
            self.prev_d_pc = d_pc
        else:
            self.prev_d_ep = d_ep
            self.prev_d_ec = d_ec

        # initialize evader algorithm parameters
        self.dash_flag = False
        self.rand_steps = 0
        self.curr_step = 0
        self.w_e_hold = 0.0

        # initialize control command tracking
        self.w_p_hist = []
        self.w_e_hist = []

        # compute the environment observation and information
        obs = self.compute_observation()

        return obs

    ## environment step ##
    def step(self, action):

        # step forward
        self.step_count += int(1)
        self.t += float(self.dt)
        action = int(action)

        # store previous positions
        prev_pos_e = self.evader_state[:2].copy()
        prev_pos_p = self.pursuer_state[:2].copy()

        # apply pursuer and evader dynamics
        w_p = self.apply_pursuer_dynamics(action)
        w_e = self.apply_evader_dynamics(action, self.step_count)
        self.w_p_hist.append(abs(float(w_p)))
        self.w_e_hist.append(abs(float(w_e)))

        # check termination conditions
        terminated, truncated, d_ec, d_ep, d_pc, evader_reached, evader_captured = self.check_termination(prev_pos_e, prev_pos_p)

        # compute reward
        reward = self.compute_reward(evader_reached, evader_captured, d_ec, d_ep, d_pc)

        # compute new observation
        obs = self.compute_observation()

        return obs, reward, terminated, truncated


    ### <===== helper functions =====> ###

    # apply pursuer dynamics (learned or default)
    def apply_pursuer_dynamics(self, action):
        x_p, y_p, theta_p, v_p = self.pursuer_state
        x_e, y_e, theta_e, v_e = self.evader_state

        # update pursuer state (learned)
        if self.controlled_agent == "pursuer":
            
            # policy control commands
            if action == 0:     # left
                w_p = self.w_p_max 
            elif action == 1:   # straight
                w_p = 0.0
            elif action == 2:   # right
                w_p = -self.w_p_max
            else:
                raise ValueError(f"Invalid action {action} encountered for Discrete(3) action space.")

        # update pursuer state (default: constant-bearing pursuit)
        else:

            # constant-bearing model
            if self.pursuer_algo == 'constant-bearing':
                LOS_angle = np.arctan2(y_e - y_p, x_e - x_p)                             # line of sight to evader
                beta = self.wrap_angle(theta_e - LOS_angle)                              # relative heading of evader wrt LOS
                alpha = np.arcsin(np.clip(v_e/(v_p + 1e-8) * np.sin(beta), -1.0, 1.0))   # lead angle
                heading_error = self.wrap_angle(LOS_angle + alpha - theta_p)             # pursuer heading error from evader
                w_p_cmd = self.k_pursuer * heading_error                                 # pursuer turn control command
                w_p = np.clip(w_p_cmd, -self.w_p_max, self.w_p_max)                      # bound pursuer turn rate

            # deviated pursuit (lead)
            elif self.pursuer_algo == 'deviated':
                X_E_pred = v_e * np.array([np.cos(theta_e), np.sin(theta_e)]) * self.T_horizon + np.array([x_e, y_e])             # predicted future position of evader
                LOS_angle = np.arctan2(X_E_pred[1] - y_p, X_E_pred[0] - x_p)                                                      # line of sight to predicted evader position
                heading_error = self.wrap_angle(LOS_angle - theta_p)                                                              # pursuer heading error from predicted evader position
                w_p_cmd = self.k_pursuer * heading_error                                                                          # pursuer turn control command
                w_p = np.clip(w_p_cmd, -self.w_p_max, self.w_p_max)                                                               # bound pursuer turn rate
            
            # homing / pure-pursuit model
            else:
                LOS_angle = np.arctan2(y_e - y_p, x_e - x_p)                             # line of sight to evader
                heading_error = self.wrap_angle(LOS_angle - theta_p)                     # pursuer heading error from evader
                w_p_cmd = self.k_pursuer * heading_error                                 # pursuer turn control command
                w_p = np.clip(w_p_cmd, -self.w_p_max, self.w_p_max)                      # bound pursuer turn rate

        # update pursuer state
        theta_p = self.wrap_angle(theta_p + self.dt * w_p)
        x_p = x_p + v_p * np.cos(theta_p) * self.dt
        y_p = y_p + v_p * np.sin(theta_p) * self.dt
        self.pursuer_state = np.array([x_p, y_p, theta_p, v_p], dtype=np.float32)

        return w_p * self.dt

    # apply evader dynamics (learned or default)
    def apply_evader_dynamics(self, action, curr_step_count):
        x_e, y_e, theta_e, v_e = self.evader_state
        x_c, y_c = self.command_center
        x_p, y_p, theta_p, v_p = self.pursuer_state

        # update evader state (learned)
        if self.controlled_agent == "evader":

            # policy control commands
            if action == 0:     # left
                w_e = self.w_e_max 
            elif action == 1:   # straight
                w_e = 0.0
            elif action == 2:   # right
                w_e = -self.w_e_max
            else:
                raise ValueError(f"Invalid action {action} encountered for Discrete(3) action space.")

        # update evader state (default: target command center)
        else:

            # alpha-blend model
            if self.evader_algo == 'alpha-blend':
                alpha = 0.5
                angle2goal = np.arctan2(y_c - y_e, x_c - x_e)                 # line of sight to command center
                LOS_p = np.arctan2(y_e - y_p, x_e - x_p)                      # bearing from pursuer to evader
                theta_perp = self.wrap_angle(LOS_p + np.pi/2.0)               # move perpendicular to pursuer LOS
                v_goal = np.array([np.cos(angle2goal), np.sin(angle2goal)])   # velocity targeting command center
                v_avoid = np.array([np.cos(theta_perp), np.sin(theta_perp)])  # velocity perpendicular to pursuer LOS
                v_blend = (1.0 - alpha)*v_goal + alpha*v_avoid                # blended velocities
                heading_error = self.wrap_angle(np.arctan2(v_blend[1], v_blend[0]) - theta_e)          # evader heading error from command center
                w_e_cmd = self.k_evader * heading_error                       # evader turn control command
                w_e = np.clip(w_e_cmd, -self.w_e_max, self.w_e_max)           # bound evader turn rate

            # random model
            elif self.evader_algo == 'random':
                d_ec = np.hypot(x_c - x_e, y_c - y_e)                                       # distance of evader to command center
                d_pc = np.hypot(x_c - x_p, y_c - y_p)                                       # distance of pursuer to command center
                evader_time2reach  = d_ec / max(v_e, 1e-8)                                  # time for evader to reach the command center
                pursuer_time2reach = d_pc / max(v_p, 1e-8)                                  # time for evader to reach the command center
                if (evader_time2reach < pursuer_time2reach) or self.dash_flag==True:        # if the evader can beat the pursuer to the command center ...
                    self.dash_flag = True                                                   # set the dash flag true if it hasnt been set already
                    angle2goal = np.arctan2(y_c - y_e, x_c - x_e)                           # line of sight to command center
                    v_goal = np.array([np.cos(angle2goal), np.sin(angle2goal)])             # velocity targeting command center
                    heading_error = self.wrap_angle(np.arctan2(v_goal[1], v_goal[0]) - theta_e)          # evader heading error from command center
                    w_e_cmd = self.k_evader * heading_error                                 # evader turn control command
                    w_e = np.clip(w_e_cmd, -self.w_e_max, self.w_e_max)                     # bound evader turn rate
                else:                                                                       # if the evader cannot beat the pursuer to the command center ...
                    w_e = self.random.uniform(-self.w_e_max, self.w_e_max)                  # take a random action

            # long random model
            elif self.evader_algo == 'long random':
                d_ec = np.hypot(x_c - x_e, y_c - y_e)                                       # distance of evader to command center
                d_pc = np.hypot(x_c - x_p, y_c - y_p)                                       # distance of pursuer to command center
                evader_time2reach  = d_ec / max(v_e, 1e-8)                                  # time for evader to reach the command center
                pursuer_time2reach = d_pc / max(v_p, 1e-8)                                  # time for evader to reach the command center
                
                if (evader_time2reach < pursuer_time2reach) or self.dash_flag:        # if the evader can beat the pursuer to the command center ...
                    self.dash_flag = True                                                   # set the dash flag true if it hasnt been set already
                    angle2goal = np.arctan2(y_c - y_e, x_c - x_e)
                    heading_error = self.wrap_angle(angle2goal - theta_e)
                    w_e_cmd = self.k_evader * heading_error                                 # evader turn control command
                    w_e = np.clip(w_e_cmd, -self.w_e_max, self.w_e_max)                     # bound evader turn rate
                else:
                    if (curr_step_count >= self.curr_step + self.rand_steps) or (curr_step_count == 0):
                        self.rand_steps = self.random.integers(10, 20)
                        self.curr_step = curr_step_count
                        self.w_e_hold = self.random.uniform(-self.w_e_max, self.w_e_max)
                    w_e = self.w_e_hold
            
            # homing model
            else:
                angle2goal = np.arctan2(y_c - y_e, x_c - x_e)                 # line of sight to command center
                v_goal = np.array([np.cos(angle2goal), np.sin(angle2goal)])   # velocity targeting command center
                heading_error = self.wrap_angle(np.arctan2(v_goal[1], v_goal[0]) - theta_e)          # evader heading error from command center
                w_e_cmd = self.k_evader * heading_error                       # evader turn control command
                w_e = np.clip(w_e_cmd, -self.w_e_max, self.w_e_max)           # bound evader turn rate

        # update the state
        theta_e = self.wrap_angle(theta_e + self.dt * w_e)
        x_e = x_e + v_e * np.cos(theta_e) * self.dt
        y_e = y_e + v_e * np.sin(theta_e) * self.dt
        self.evader_state = np.array([x_e, y_e, theta_e, v_e], dtype=np.float32)

        return w_e * self.dt

    # function to compute observation state for controlled agent
    def compute_observation(self):
        if self.controlled_agent == "pursuer":
            obs = np.concatenate([self.pursuer_state, self.evader_state]).astype(np.float32)
        else:
            obs = np.concatenate([self.evader_state, self.pursuer_state]).astype(np.float32)
        return obs

    # function to compute the agent's reward at the current time step
    def compute_reward(self, evader_reached, evader_captured, d_ec, d_ep, d_pc):
        reward = 0.0

        # reward the controlled agent (pursuer)
        if self.controlled_agent == "pursuer":

            # check distance to evader
            if d_ep < self.prev_d_ep:
                reward += self.reward_d_ep
            self.prev_d_ep = d_ep

            # check distance to command center
            if d_pc > self.prev_d_pc:
                reward += self.reward_d_pc
            self.prev_d_pc = d_pc

            # check termination
            if evader_captured:
                reward += self.reward_success
            elif evader_reached:
                reward -= self.reward_failure

        # reward the controlled agent (evader)
        else:

            # check distance to command center
            if d_ec < self.prev_d_ec:
                reward += self.reward_d_ec
            self.prev_d_ec = d_ec

            # check distance to pursuer
            if d_ep > self.prev_d_ep:
                reward += self.reward_d_ep
            self.prev_d_ep = d_ep

            # check termination
            if evader_reached:
                reward += self.reward_success
            elif evader_captured:
                reward -= self.reward_failure

        return reward

    # function to check success, capture, or termination conditions
    def check_termination(self, prev_evader_pos, prev_pursuer_pos):
        x_e, y_e, _, _ = self.evader_state
        x_p, y_p, _, _ = self.pursuer_state
        x_c, y_c = self.command_center

        # compute current agent distances
        d_ec = np.hypot(x_e - x_c, y_e - y_c)
        d_ep = np.hypot(x_e - x_p, y_e - y_p)
        d_pc = np.hypot(x_p - x_c, y_p - y_c)

        # determine if agents have reached their targets
        evader_reached_endpoint = (d_ec <= self.command_center_radius)
        evader_reached_segment  = self.check_continuous_collision(prev_evader_pos, np.array([x_e, y_e], dtype=np.float32), self.command_center, self.command_center_radius)
        evader_reached = evader_reached_endpoint or evader_reached_segment

        evader_captured_endpoint = (d_ep <= self.evader_radius)
        evader_captured_segment  = self.check_continuous_collision(prev_pursuer_pos, np.array([x_p, y_p], dtype=np.float32), np.array([x_e, y_e], dtype=np.float32), self.evader_radius)
        evader_captured = evader_captured_endpoint or evader_captured_segment

        # compute termination conditions
        terminated = evader_reached or evader_captured
        truncated = (self.step_count >= self.max_steps) and not terminated

        return terminated, truncated, d_ec, d_ep, d_pc, evader_reached, evader_captured
    
    # function to wrap angles [-pi, +pi]
    def wrap_angle(self, theta):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi
    
    # function to compute continuous collision
    def check_continuous_collision(self, p0, p1, c, r):

        # establish the start, end, and collision points
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        c  = np.asarray(c,  dtype=np.float32)

        # compute the lines
        d = p1 - p0
        f = p0 - c

        # verify that line segment is valid
        a = np.dot(d, d)
        if a == 0.0:
            return np.dot(f, f) <= r**2

        # check for intersection
        b = 2.0 * np.dot(f, d)
        c_val = np.dot(f, f) - r**2
        discriminant = b*b - 4*a*c_val
        if discriminant < 0.0:
            return False
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)
    
    # function to return integrated control history
    def get_integrated_control_histories(self):
        return np.asarray(self.w_p_hist, dtype=np.float32), np.asarray(self.w_e_hist, dtype=np.float32)
    
    # function to visualize the current game
    def render(self, show=True):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect("equal")
        self.ax.clear()

        # get environment states
        x_c, y_c = self.command_center
        x_e, y_e, theta_e, _ = self.evader_state
        x_p, y_p, theta_p, _ = self.pursuer_state

        # plot command center
        self.ax.scatter(x_c, y_c, marker="s", s=80, label="Command Center", edgecolors="k")

        # plot evader and pursuer
        self.ax.scatter(x_e, y_e, marker="o", s=50, label="Evader")
        self.ax.scatter(x_p, y_p, marker="^", s=50, label="Pursuer")

        # draw small heading arrows for each agent
        arrow_scale = 5.0
        self.ax.arrow(
            x_e, y_e,
            arrow_scale * np.cos(theta_e),
            arrow_scale * np.sin(theta_e),
            head_width=2.0,
            length_includes_head=True,
            alpha=0.7,
        )
        self.ax.arrow(
            x_p, y_p,
            arrow_scale * np.cos(theta_p),
            arrow_scale * np.sin(theta_p),
            head_width=2.0,
            length_includes_head=True,
            alpha=0.7,
        )

        radius = 50.0
        self.ax.set_xlim(x_c - radius, x_c + radius)
        self.ax.set_ylim(y_c - radius, y_c + radius)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_title(f"Pursuit–Evasion (t = {self.t:.2f} s)")
        self.ax.legend(loc="upper right")

        self.fig.canvas.draw()
        if show:
            plt.pause(0.001)

    # function to clean up and close the environment
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None




