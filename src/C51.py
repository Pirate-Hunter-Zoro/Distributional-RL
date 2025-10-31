from turtle import update
import gymnasium as gym
import numpy as np
from src.utils.envs import *
from src.utils.seed import *
from src.utils.buffers import *
from src.utils.torch import *
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Suggested constants
ATOMS = 51              # Number of atoms for distributional network
ZRANGE = [0, 200]       # Range for Z projection
z = torch.linspace(ZRANGE[0], ZRANGE[1], ATOMS).to(DEVICE)
delta_z = (ZRANGE[1] - ZRANGE[0]) / (ATOMS - 1)

# Global variables
EPSILON = STARTING_EPSILON
Z = None

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    create_seed(seed)
    env = TimeLimit(NoisyCartPole(), 500)
    test_env = TimeLimit(NoisyCartPole(), 500)
    buf = ReplayBuffer(BUFSIZE)
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, EPSILON_END, STEPS_MAX, Z, z
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        ## Each action has a distribution of expected values
        action_atom_logits = Z(obs).reshape((ACT_N, ATOMS))
        # Each action has a vector of 'value' positions - we have yet to fill those positions with the actual values associated
        # BUT first we need the probabilities of said positions, so use softmax on the logits
        action_atom_probs = torch.softmax(action_atom_logits, dim=1) # ACT_N x ATOMS
        # Multiply the probabilities by the atoms by their respective z-values so now each atom-value is weighted by its expected probability for EACH action
        weighted_atom_values = action_atom_probs * z # ACT_N x ATOMS
        # Find expected q-value for each action by summing each atom value weighted by said atom's probability (which we just computed)
        expected_q_values = torch.sum(weighted_atom_values, dim=1) # ACT_N x 1
        # Pick the action with the highest resulting expected reward
        action = torch.argmax(expected_q_values, dim=0).item() # integer
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# Update networks
def update_networks(epi, buf, Z, Zt, OPT):
    global z, delta_z, GAMMA, ZRANGE, MINIBATCH_SIZE, t
    
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    # The same kind of creature we dealt with above in the 'policy' function except it's over an entire batch of states
    target_atom_prob_logits = Zt(S2).reshape(MINIBATCH_SIZE, ACT_N, ATOMS)
    # Softmax to turn to probability distribution over atoms
    target_atom_probs = torch.softmax(target_atom_prob_logits, dim=2) # MINIBATCH_SIZE x ACT_N x ATOMS
    # Weight atom values by their probabilies
    target_actions_weighted_atom_values = target_atom_probs * z # SAME DIM SO FAR
    # Weighted sum of atom values by their probabilities
    target_q_values = torch.sum(target_actions_weighted_atom_values, dim=2) # weighted sum of all atom values - MINIBATCH_SIZE x ACT_N x 1
    # For each step in the batch, find the action with the best q_value
    target_next_actions = torch.argmax(target_q_values, dim=1) # for each state in the minibatch, find the next best action - MINIBATCH_SIZE x 1
    # Gather the atom probabilities of these next actions to be taken
    target_taken_actions_probs = torch.gather(target_atom_probs, dim=1, index=target_next_actions.view(MINIBATCH_SIZE, 1, 1).expand((MINIBATCH_SIZE, 1, ATOMS))).squeeze() # MINIBATCH_SIZE x ATOMS
    
    # For each action taken in each step of our minibatch, what are the atom probabilities?
    target_distribution = torch.zeros((MINIBATCH_SIZE, ATOMS))
    for j in range(ATOMS):
        # Current reward plus discounted value of said atom (for EACH step in the minibatch)
        projected_atom_j = R + GAMMA * z[j] * (1-D) # MINIBATCH_SIZE x 1 (projection happens here)
        projected_atom_j = torch.clip(projected_atom_j, ZRANGE[0], ZRANGE[1])
        # For each step in the batch that may not be one of our discrete atom values - find the nearest one that does not exceed it
        b_float = (projected_atom_j - ZRANGE[0])/delta_z # MINIBATCH_SIZE x 1
        b_lower = b_float.floor().long() # For each step in the batch, this is the index of the atom closest to the projected atom value without exceeding
        b_lower = b_lower.clamp(0, ATOMS-1)
        b_upper = b_float.ceil().long() # Same but closest without going below
        b_upper = b_upper.clamp(0, ATOMS-1)
        weight_upper_bin = b_float - b_lower.float() # MINIBATCH_SIZE x 1
        weight_lower_bin = 1 - weight_upper_bin # MINIBATCH_SIZE x 1
        # Now grab the probability of this atom over all steps in the batch
        taken_action_atom_prob = target_taken_actions_probs[:,j] # MINIBATCH_SIZE x 1
        # Weight the probabilities of the upper and lower bins - since they were closest to the reward we achieved so their probabilities should go up - according to the previous atom's projected value
        target_distribution.scatter_add(dim=1, index=b_lower.unsqueeze(1), src=(weight_lower_bin*taken_action_atom_prob).unsqueeze(1))
        target_distribution.scatter_add(dim=1, index=b_upper.unsqueeze(1), src=(weight_upper_bin*taken_action_atom_prob).unsqueeze(1))
    
    # Now that we have the target distribution of atom probabilities, we can compute the loss
    actions_atom_logits = Z(S).reshape(MINIBATCH_SIZE, ACT_N, ATOMS) # BATCH_SIZE x ACT_N x ATOMS
    # Only grab the logits for the actions we care about
    taken_actions_atom_logits = torch.gather(actions_atom_logits, dim=1, index=A.view(MINIBATCH_SIZE, 1, 1).expand((MINIBATCH_SIZE, 1, ATOMS))).squeeze() # BATCH_SIZE x ATOMS
    # Turn those logits into log probabilities
    log_preds = torch.log_softmax(taken_actions_atom_logits, dim=1)
    # The closer the target_distribution agrees with the log_preds, the happier we are
    loss = -(target_distribution.detach() * log_preds).sum(dim=1).mean()
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # Update target network
    if epi%TARGET_NETWORK_UPDATE_FREQ==0:
        Zt.load_state_dict(Z.state_dict())

    return loss


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Z, Zt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    plt.show()