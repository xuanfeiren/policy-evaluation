import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3 import PPO
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make("CartPole-v1")

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
        return self.layer3(x)
    
policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)

# Load the trained model weights
policy_net.load_state_dict(torch.load("policy_net.pth", weights_only=True))


def policy_DQN(state):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return policy_net(state).max(1).indices.view(1, 1)
    

def policy_uniform(state):
    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# num_episodes = 10

# for i_episode in range(num_episodes):
#     state, _ = env.reset()
#     for t in range(1, 10000):
#         action = policy_DQN(state)
#         state, reward, terminated, truncated, _ = env.step(action.item())
#         if terminated or truncated:
#             print(f"DQN policy: Episode {i_episode + 1} finished after {t} timesteps")
#             break

# for i_episode in range(num_episodes):
#     state, _ = env.reset()
#     for t in range(1, 10000):
#         action = policy_uniform(state)
#         state, reward, terminated, truncated, _ = env.step(action.item())
#         if terminated or truncated:
#             print(f"Uniform policy: Episode {i_episode + 1} finished after {t} timesteps")
#             break

# model_PPO = PPO.load("ppo_cartpole")
# def policy_PPO(s):
#   action = model_PPO.predict(s)[0]
#   return action

# for i_episode in range(num_episodes):
#     state, _ = env.reset()
#     for t in range(1, 10000):
#         action = policy_PPO(state)
#         state, reward, terminated, truncated, _ = env.step(action.item())
#         if terminated or truncated:
#             print(f"PPO policy: Episode {i_episode + 1} finished after {t} timesteps")
#             break

# def policy_mix(state, epsilon=0.4):
#     if random.random() < epsilon:
#         return policy_uniform(state)
#     else:
#         return policy_DQN(state)

# for i_episode in range(num_episodes):
#     state, _ = env.reset()
#     for t in range(1, 10000):
#         action = policy_mix(state)
#         state, reward, terminated, truncated, _ = env.step(action.item())
#         if terminated or truncated:
#             print(f"Mix policy: Episode {i_episode + 1} finished after {t} timesteps")
#             break

# Print some policy_DQN values for random states
for _ in range(5):
    state = env.observation_space.sample()
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_values = policy_net(state)
    print(f"State: {state}, DQN Action Values: {action_values}")