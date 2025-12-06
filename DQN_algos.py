#######################################################################################
# Filename: DQN_algos.py
#
# Description: This script contains functions for performing the DQN learning method
#              on a given MDP.
#
# References:
#   Sutton&Barto.pdf: file:///C:/Users/wsche/Documents/AE%20598/Sutton&Barto.pdf
#   Mnih et al.: https://www.nature.com/articles/nature14236
#
#######################################################################################
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

## HELPER FUNCTIONS ##

# function to encode the current state as a flattened array
def encode_state(s):
    s_np = np.array(s)
    return s_np.astype(np.float32)

# function to build the neural network
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

# function to choose the action from Q(s|a)
def getActionFromQ(Q, s, epsilon, epsilon_final, epsilon_decay_steps, nS, nA, device):

    # explore or exploit based on current epsilon value
    if np.random.rand() < epsilon:
        a = np.random.randint(nA)   # explore

    else:
        with torch.no_grad():   # exploit
            
            # encode the state as a flattened array
            s = encode_state(s)

            # convert the state into a torch tensor
            s_torch = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

            # get the action-values for the current state
            q = Q(s_torch)

            # choose the greedy action
            a = int(q.argmax(dim=1).item())
    
    # decay epsilon
    epsilon_decay = (epsilon - epsilon_final) / float(epsilon_decay_steps)
    epsilon = max(epsilon_final, epsilon - epsilon_decay)

    return a, epsilon

# function to convert transition values to torch tensors
def vals2torch(s, a, r, s_next, t, device):
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
    r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
    s_next = torch.tensor(s_next, dtype=torch.float32, device=device)
    t = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1)
    return s, a, r, s_next, t


## DQN ALGORITHM ##

# function to update Q_online
def Q_online_update(Q_online, Q_target, memory_buffer, batch_size, nS, gamma, optimizer, device):

    # skip the update if not enough memory in the buffer
    if len(memory_buffer) < batch_size:
        return Q_online, optimizer, None
    
    # sample a random batch of transitions from the memory buffer
    rand_idxs = np.random.choice(len(memory_buffer), batch_size, replace=False)
    batch = []
    for idx in rand_idxs:
        batch.append(memory_buffer[idx])

    # collect the transitions in the batch and encode the state info as a flattened array
    s, a, r, s_next, t = zip(*batch)
    s_enc = []
    sn_enc = []
    for si, sin in zip(s, s_next):
        s_enc.append(encode_state(si))
        sn_enc.append(encode_state(sin))

    # format transitions as numpy arrays
    s = np.stack(s_enc)
    s_next = np.stack(sn_enc)
    r = np.array(r, dtype=np.float32)
    a = np.array(a, dtype=np.int64)
    t = np.array(t, dtype=np.float32)

    # convert transitions to torch tensors
    s, a, r, s_next, t = vals2torch(s, a, r, s_next, t, device)

    # get the current online action values (multiple actions, each have a value) from each sampled state
    q = Q_online(s).gather(1, a)

    # compute the target term (r + gamma * max_a'(q_target(s',a'; theta_i)))
    with torch.no_grad():

        # get the action values (multiple actions, each have a value) at each of the next states
        q_next = Q_target(s_next)

        # get the action with the maximum value at each of the next states
        q_next_max, _ = q_next.max(dim=1, keepdim=True)

        # compute the target term for each of the next states
        targets = r + gamma * (1.0 - t) * q_next_max   # if the next state is terminal, t==1 and targets==r

    # compute the loss ( (targets - q_online(s,a;theta_i))^2 )
    L = nn.MSELoss()(q, targets)

    # perform gradient descent to update network weights (theta)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    L = float(L.item())

    return Q_online, optimizer, L

# function to run DQN training (default values match "Human-level control through deep reinforcement learning")
def DQN(num_episodes, 
        max_steps,
        env,
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_final=0.1, 
        epsilon_decay_steps=1000000, 
        alpha=0.00025, 
        hidden_layer_sizes=(64,64,64,64), 
        max_buffer_size=1000000, 
        batch_size=32,
        q_target_update_freq=10000,
        replay_start_size=50000
        ):
    
    # start the timer
    print('\nStarting DQN training ...')
    t_start = time.perf_counter()

    # initialize environment information
    obs_space = env.observation_space
    act_space = env.action_space
    nS = obs_space.shape[0]
    nA = act_space.n

    # initialize network values
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    Q_online = build_neural_network(nS, nA, hidden_layer_sizes).to(device)
    Q_target = build_neural_network(nS, nA, hidden_layer_sizes).to(device)
    Q_target.load_state_dict(Q_online.state_dict())
    Q_target.eval()
    optimizer = optim.Adam(Q_online.parameters(), lr=alpha)

    # initialize algorithm and evaluation values
    memory_buffer = []
    total_steps = 0
    episode_steps_list = []
    pi2eval = []
    model = {k: v.detach().cpu().clone() for k, v in Q_online.state_dict().items()}
    pi2eval.append([total_steps, model])

    # train the model
    for ep in range(num_episodes):
        if ep % 10 == 0:
            print(f'{ep} episodes finished with steps {episode_steps_list} ...')
            episode_steps_list = []

        episode_terminated = False
        episode_steps = 0
        obs = env.reset()

        for _ in range(max_steps):

            # get the epsilon-greedy action from Q_online
            a, epsilon = getActionFromQ(Q_online, obs, epsilon, epsilon_final, epsilon_decay_steps, nS, nA, device)
            episode_steps += 1
            total_steps += 1

            # step in the environment
            next_obs, reward, terminated, truncated = env.step(a)
            if terminated or truncated:
                episode_terminated = True

            # store the transition in the replay buffer
            memory_buffer.append((obs, a, float(reward), next_obs, float(episode_terminated)))
            if len(memory_buffer) > max_buffer_size:
                memory_buffer.pop(0)

            # start learning after buffer reaches the defined start size
            if len(memory_buffer) >= replay_start_size:
                Q_online, optimizer, _ = Q_online_update(Q_online, Q_target, memory_buffer, batch_size, nS, gamma, optimizer, device)
            
            # update the Q_target periodically
            if total_steps % q_target_update_freq == 0:
                Q_target.load_state_dict(Q_online.state_dict())
            
            # end the episode if terminated
            if episode_terminated:
                break

            # move to the next state
            obs = next_obs
        
        # collect the current greedy policy (end of every episode) and the cumulative sum of steps taken to get there
        model = {k: v.detach().cpu().clone() for k, v in Q_online.state_dict().items()}
        pi2eval.append([total_steps, model])
        episode_steps_list.append(episode_steps)
    
    # close the environment and end the timer
    env.close()
    total_time = time.perf_counter() - t_start
    print(f'Completed DQN training ({total_time:.2f} s)\n')
    
    return Q_online, pi2eval


