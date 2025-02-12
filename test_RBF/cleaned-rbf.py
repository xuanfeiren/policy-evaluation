# cleaned-rbf.py
# from Cart-Pole-torch-cleaned.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from itertools import count
from collections import namedtuple, deque
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
import wandb
from sklearn.kernel_approximation import RBFSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0")
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
class RBFNet(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.n_components = 500
        self.linear = nn.Linear(self.n_components, output_size)
        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.n_components)
        self.device = device
        
    def forward(self, x):
        # Convert to numpy and ensure 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if single sample
        x_np = x.cpu().numpy()
        
        # Transform features
        feature_np = self.rbf_feature.fit_transform(x_np)
        feature = torch.tensor(feature_np, dtype=torch.float).to(self.device)
        output = self.linear(feature)
        
        # If input was single sample, squeeze output
        if x_np.shape[0] == 1:
            output = output.squeeze(0)
            
        return output
        
    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.weight)
rbf_net = RBFNet(n_observations, n_actions, device).to(device)
rbf_net.load_state_dict(torch.load('../rbf_net.pth'))
rbf_net.eval()
def policy_rbf(s0):
    with torch.no_grad():
        s0 = s0[0] if isinstance(s0, tuple) else s0
        if not torch.is_tensor(s0):
            s0 = torch.tensor(s0, dtype=torch.float).to(device)
        q_values = rbf_net(s0)
        return torch.argmax(q_values).item()
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed()

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Load the trained model weights




class RBFNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.n_components = 256
        self.linear = nn.Linear(self.n_components, output_size)
        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.n_components)
        self.device = device
        
    def forward(self, x):
        # Convert to numpy and ensure 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if single sample
        x_np = x.cpu().numpy()
        
        # Transform features
        feature_np = self.rbf_feature.fit_transform(x_np)
        feature = torch.tensor(feature_np, dtype=torch.float).to(self.device)
        output = self.linear(feature)
            
        return output
        
    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.weight)




policy_net_LSTD = RBFNet(n_observations, n_actions).to(device)
target_net_LSTD = RBFNet(n_observations, n_actions).to(device)
target_net_LSTD.load_state_dict(policy_net_LSTD.state_dict())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory_size = int(1e4)
memory = ReplayMemory(memory_size)
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

optimizer_LSTD = optim.AdamW(policy_net_LSTD.parameters(), lr=LR, amsgrad=True)




episode_loss_LSTD = []

print("Evaluating testing states...")
num_test_states = 100
states_list = []
for _ in range(num_test_states):
    state, _ = env.reset()
    states_list.append(state)
states_array = np.array(states_list)
states = torch.tensor(states_array, dtype=torch.float32, device=device)

def calculate_mc_return(state, num_trajectories=1, max_steps=400):
    '''environment and policy are deterministic, so one trajectory is enough'''
    n_actions = env.action_space.n  # CartPole action space
    total_returns = np.zeros(n_actions)
    
    for action_idx in range(n_actions):
        action_return = 0
        for _ in range(num_trajectories):
            s0 , _ = env.reset()
            trajectory_return = 0
            # a = env.unwrapped.state
            env.unwrapped.state = state
            # b = env.unwrapped.state
            current_state = state
            
            # First action is fixed
            next_state, reward, terminated, truncated, _ = env.step(action_idx)
            trajectory_return += reward
            
            # Continue with DQN policy after first step
            if not (terminated or truncated):
                current_state = next_state
                for step in range(1, max_steps):
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
                    
                    next_action = policy_rbf(state_tensor)
                    next_state, reward, terminated, truncated, _ = env.step(next_action)
                    trajectory_return += GAMMA ** step * reward
                    
                    if terminated or truncated:
                        break
                    current_state = next_state
            
            action_return += trajectory_return
        
        total_returns[action_idx] = action_return / num_trajectories
    
    return torch.tensor(total_returns, dtype=torch.float32, device=device)

# Calculate expected returns for all initial states
expected_returns = []
for state in states_list:
    mc_return = calculate_mc_return(state)  # Returns 2D vector
    expected_returns.append(mc_return)
expected_returns = torch.stack(expected_returns)  # Shape: [100, 2]
print("Done evaluating testing states.")

def calculate_loss(policy_net):
    # Calculate the loss
    total_loss = 0
    with torch.no_grad():
        for state, mc_return in zip(states, expected_returns):
            state = state.unsqueeze(0)  # Add batch dimension
            policy_output = policy_net(state)
            loss = F.mse_loss(policy_output.squeeze(0), mc_return)
            total_loss += loss
    return total_loss / num_test_states

TAU = 1

num_collect_data = 40
print("Collecting data...")
for i_collect_data in tqdm(range(num_collect_data)):
    '''data collecting process'''
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = policy_rbf(state) # on policy evaluation
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state
        if done:
            break
print("Done collecting", len(memory), " data.")

print("Training...")
num_episodes = int(1e5)
wandb.init(
    # set the wandb project where this run will be logged
    project="rfflow",

    # track hyperparameters and run metadata
    config={
    "num_episodes": num_episodes,
    }
)
def optimize_models():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = Transition(*zip(*transitions))

    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # state_batch = torch.cat([torch.tensor([s], device=device) for s in batch.state])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat([torch.tensor([a], device=device) for a in batch.action])
    reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float) for r in batch.reward])  

    # action_batch = torch.cat(batch.action)
    # reward_batch = torch.cat(batch.reward)

    '''Optimizer LSTD model'''

    state_action_values_LSTD = policy_net_LSTD(state_batch).gather(1, action_batch.unsqueeze(1))
    with torch.no_grad():
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_actions = torch.tensor([
            policy_rbf(state.unsqueeze(0))
            for state in non_final_next_states
        ], device=device).unsqueeze(1)
        next_q_values = target_net_LSTD(non_final_next_states)
        next_state_values[non_final_mask] = next_q_values.gather(1, next_actions).squeeze()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.MSELoss()
    loss_LSTD = criterion(state_action_values_LSTD, expected_state_action_values.unsqueeze(1))

    # '''Optimize LSTD model'''
    # state_action_values_LSTD = policy_net_LSTD(state_batch).gather(1, action_batch)
    # next_state_values_LSTD = torch.zeros(BATCH_SIZE, device=device)
    # with torch.no_grad():
    #     next_state_values_LSTD[non_final_mask] = target_net_LSTD(non_final_next_states).max(1).values #in dqn
        
    # # Get next actions using policy_DQN
    # expected_state_action_values_LSTD = (next_state_values_LSTD * GAMMA) + reward_batch

    # criterion = nn.MSELoss()
    # loss_LSTD = criterion(state_action_values_LSTD, expected_state_action_values_LSTD.unsqueeze(1))


    # Optimize the model
    optimizer_LSTD.zero_grad()
    loss_LSTD.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_LSTD.parameters(), 100)
    optimizer_LSTD.step()
for i_episode in tqdm(range(num_episodes)):
    optimize_models()
    target_net_state_dict = target_net_LSTD.state_dict()
    policy_net_state_dict = policy_net_LSTD.state_dict()
    for key in policy_net_state_dict:
        # target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net_state_dict[key] = policy_net_state_dict[key]
    target_net_LSTD.load_state_dict(target_net_state_dict)
    if i_episode % 100 == 0:
        loss_LSTD = calculate_loss(policy_net_LSTD).item()
        episode_loss_LSTD.append(loss_LSTD)
        wandb.log({
            "Episode": i_episode,
            "Loss/LSTD": loss_LSTD,
        })
wandb.finish()