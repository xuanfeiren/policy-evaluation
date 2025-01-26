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
import matplotlib
import matplotlib.pyplot as plt
import wandb
from sklearn.kernel_approximation import RBFSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")

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
DQN_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)

# Load the trained model weights
DQN_net.load_state_dict(torch.load("policy_net.pth", weights_only=True))


def policy_DQN(state):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    actions = DQN_net(state)
    action = actions.max(1)[1].view(1, 1)
    return action

class FeatureExtractor(nn.Module):
    def __init__(self, dqn_net):
        super(FeatureExtractor, self).__init__()
        # Copy first two layers and ReLU
        self.layer1 = dqn_net.layer1
        self.layer2 = dqn_net.layer2
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x

feature_mapping = FeatureExtractor(DQN_net)
class Linear_model(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Linear_model, self).__init__()
        # Copy all layers except last from DQN_net
        self.feature_mapping = feature_mapping
        # Freeze feature mapping weights
        for param in self.feature_mapping.parameters():
            param.requires_grad = False
            
        # Get feature dimension from last layer
        feature_dim = 128
        self.linear = nn.Linear(feature_dim, n_actions, bias=False)

    def forward(self, x):
        x = self.feature_mapping(x)
        return self.linear(x)
    
    def init(self):
        nn.init.zeros_(self.linear.weight)

class RBFNet(nn.Module):
    def __init__(self, input_size, output_size):
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
        # if x_np.shape[0] == 1:
        #     output = output.squeeze(0)
            
        return output
        
    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.weight)

n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# policy_net_BRM = Linear_model(n_observations, n_actions).to(device)
# policy_net_LSTD = Linear_model(n_observations, n_actions).to(device)
# target_net_LSTD = Linear_model(n_observations, n_actions).to(device)
policy_net_BRM = RBFNet(n_observations, n_actions).to(device)
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

memory = ReplayMemory(10000)
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

optimizer_BRM = optim.AdamW(policy_net_BRM.parameters(), lr=LR, amsgrad=True)
optimizer_LSTD = optim.AdamW(policy_net_LSTD.parameters(), lr=LR, amsgrad=True)

def optimize_models():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = Transition(*zip(*transitions))

    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    '''Optimizer BRM model'''
    state_action_values = policy_net_BRM(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_actions = torch.tensor([
        policy_DQN(state.unsqueeze(0)).item() 
        for state in non_final_next_states
    ], device=device).unsqueeze(1)
    next_q_values = policy_net_BRM(non_final_next_states)
    next_state_values[non_final_mask] = next_q_values.gather(1, next_actions).squeeze()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.MSELoss()
    loss_BRM = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer_BRM.zero_grad()
    loss_BRM.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_BRM.parameters(), 100)
    optimizer_BRM.step()

    '''Optimize LSTD model'''
    state_action_values_LSTD = policy_net_LSTD(state_batch).gather(1, action_batch)
    next_state_values_LSTD = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values_LSTD[non_final_mask] = target_net_LSTD(non_final_next_states).max(1).values #in dqn
        
    # Get next actions using policy_DQN
    expected_state_action_values_LSTD = (next_state_values_LSTD * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss_LSTD = criterion(state_action_values_LSTD, expected_state_action_values_LSTD.unsqueeze(1))


    # Optimize the model
    optimizer_LSTD.zero_grad()
    loss_LSTD.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_LSTD.parameters(), 100)
    optimizer_LSTD.step()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_loss_BRM = []
episode_loss_LSTD = []

def plot_loss(show_result=False):
    plt.figure(1)
    loss_t_BRM = torch.tensor(episode_loss_BRM, dtype=torch.float)
    loss_t_LSTD = torch.tensor(episode_loss_LSTD, dtype=torch.float)
    plt.title('CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(loss_t_BRM.numpy())
    plt.plot(loss_t_LSTD.numpy())


states_list = []
for _ in range(100):
    state, _ = env.reset()
    states_list.append(state)
states = torch.tensor(states_list, dtype=torch.float32, device=device)

def calculate_loss(policy_net, DQN_net):
    # Calculate the loss
    total_loss = 0
    with torch.no_grad():
        for state in states:
            # Process single state
            state = state.unsqueeze(0)  # Add batch dimension
            policy_output = policy_net(state)
            dqn_output = DQN_net(state)
            
            # Accumulate loss
            loss = F.mse_loss(policy_output, dqn_output)
            total_loss += loss
            
    
    return total_loss / 100

num_episodes = 600

wandb.init(
    # set the wandb project where this run will be logged
    project="test_RBF",

    # track hyperparameters and run metadata
    config={
    "num_episodes": num_episodes,
    }
)
TAU = 1
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = policy_DQN(state) # on policy evaluation
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_models()
        target_net_state_dict = target_net_LSTD.state_dict()
        policy_net_state_dict = policy_net_LSTD.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net_LSTD.load_state_dict(target_net_state_dict)
        if done:
            loss_BRM = calculate_loss(policy_net_BRM, DQN_net).item()
            loss_LSTD = calculate_loss(policy_net_LSTD, DQN_net).item()
            episode_loss_BRM.append(loss_BRM)
            episode_loss_LSTD.append(loss_LSTD)
            

            wandb.log({
               "Episode": i_episode,
                "Loss/BRM": loss_BRM,
                "Loss/LSTD": loss_LSTD,
            })
            
            break
    
print('Complete')
plot_loss(show_result=True)
plt.ioff()
plt.show()
# plt.savefig("loss_CartPole-v1.png")
wandb.finish()