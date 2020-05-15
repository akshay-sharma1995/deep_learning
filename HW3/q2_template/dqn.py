import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The starter code follows the tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# we recommend you going through the tutorial before implement DQN algorithm


# define environment, please don't change 
env = gym.make('CartPole-v1')

# define transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    define replay buffer class
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay_buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.replay_buffer.append(transition)

    def sample(self, batch_size):

        sampled_ids = np.random.choice(np.arange(len(self.replay_buffer)), size=batch_size)
        sampled_transitions = [self.replay_buffer[id] for id in sampled_ids]
        return np.array(sampled_transitions)

    def __len__(self):
        return len(self.replay_buffer)


class DQN(nn.Module):
    """
    build your DQN model:
    given the state, output the possiblity of actions
    """
    def __init__(self, in_dim, out_dim):
        """
        in_dim: dimension of states
        out_dim: dimension of actions
        """
        super(DQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(nn.Linear(self.in_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.out_dim))
        
        self.initialize_params()

    def forward(self, x):
        # forward pass
        # pdb.set_trace()
        out = self.model(x)
        
        return out
    
    def initialize_params(self):
        for param in self.parameters():
            if(len(param.shape)>1):
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.constant_(param, 0.0)

# hyper parameters you can play with
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.04
EPS_DECAY = 0.0095
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

policy_net = DQN(n_states, n_actions).to(DEVICE)
target_net = DQN(n_states, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0
epsilon = 1.00
def select_action(state):
    # given state, return the action with highest probability on the prediction of DQN model
    # you are recommended to also implement a soft-greedy here
    with torch.no_grad():
        q_vals = policy_net(state)
    q_vals = q_vals.cpu().numpy() 
    if(random.random() > epsilon):
        action = np.argmax(q_vals)
    else:
        action = np.random.choice(q_vals.shape[1], size=1)[0]

    return action

def create_actions_mask(actions):
    action_mask = np.zeros((actions.shape[0], n_actions))
    for id, mask in enumerate(action_mask):
        mask[actions[id]] = 1
    return action_mask
    
def optimize_model():
    # optimize the DQN model by sampling a batch from replay buffer
    if len(memory)<BATCH_SIZE:
        return

    sampled_transitions = memory.sample(BATCH_SIZE)
    # sampled_transitions = Transition(*zip(*sampled_transitions))

    X_train = torch.tensor([transition[0] for transition in sampled_transitions]).float().to(DEVICE)
    transition_actions = np.array([transition[1] for transition in sampled_transitions])
    action_mask = torch.tensor(create_actions_mask(transition_actions), dtype=torch.bool).to(DEVICE)
    exp_rewards = torch.tensor([transition[3] for transition in sampled_transitions]).float().to(DEVICE)
    sampled_nxt_states = np.array([transition[2] for transition in sampled_transitions])
    dones = np.array([int(transition[4]) for transition in sampled_transitions])
    
    with torch.no_grad():
        q_max_nxt_state, _ = torch.max(target_net(torch.from_numpy(sampled_nxt_states).float().to(DEVICE)), axis=1)
    
    q_vals_target = exp_rewards + GAMMA*q_max_nxt_state*torch.tensor(1-dones).float().to(DEVICE)

    Y_pred_all_actions = policy_net(X_train)
    Y_pred = torch.masked_select(Y_pred_all_actions, action_mask)

    loss = F.mse_loss(Y_pred, q_vals_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    



num_episodes = 100
episode_durations = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    epsilon = max((EPS_START - EPS_DECAY*i_episode),EPS_END)
    state = env.reset()
    # state = torch.from_numpy(state).float().view(1, -1)
    for t in count():
        # Select and perform an action

        action = select_action(torch.unsqueeze(torch.from_numpy(state), dim=0).float().to(DEVICE))
        new_state, reward, done, _ = env.step(action)

        memory.push((state, action, new_state*1.0, reward, done))
        # # Observe new state
        # if not done:
        # else:
            # next_state = None

        # Store the transition in memory

        # Move to the next state
        state = new_state*1.0

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            print("Episode: {}, duration: {}".format(i_episode, t+1))
            break
    
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save({"policy_net": policy_net.state_dict(),}, "p2_model.ckpt")

# plot time duration
plt.figure()
plt.plot(np.arange(len(episode_durations)), episode_durations)
plt.savefig("training_performance.png")
plt.show()
plt.close()

test_ep_perfomance = []
# visualize 
for i in range(10):
    state = env.reset()
    for t in count():
        env.render()

        # Select and perform an action
        action = select_action(torch.unsqueeze(torch.from_numpy(state), dim=0).float().to(DEVICE))
        new_state, reward, done, _ = env.step(action)
        reward = torch.tensor([reward])

        # Observe new state
        state = new_state*1.0

        if done:
            test_ep_perfomance.append(t + 1)
            print("Duration:", t+1)
            break
print("mean_test_episode_duration: ", np.mean(test_ep_perfomance))
# plt.figure()
# plt.scatter(np.arange(len(episode_durations)), episode_durations)
# plt.savefig("training_performance.png")
# plt.show()

env.close()
