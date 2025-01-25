import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

print(device)

from sklearn.kernel_approximation import RBFSampler

X = np.array([[0, 1, 1]])

# print(X.shape)

rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=100)
X_features = rbf_feature.fit_transform(X)

# print(X_features)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device = device
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.weight)


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

class DQN_Agent(object):
    def __init__(self, **kwargs):
        self.gamma = kwargs['gamma']
        self.epsi_high = kwargs['epsi_high'] 
        self.epsi_low = kwargs['epsi_low']
        self.decay = kwargs['decay']
        self.lr = kwargs['lr']
        self.buffer = []  # Replay buffer
        self.capacity = kwargs['capacity']
        self.batch_size = kwargs['batch_size']
        self.state_space_dim = kwargs['state_space_dim']
        self.action_space_dim = kwargs['action_space_dim']
        self.device = kwargs['device']
        self.network_type = kwargs['network_type']  # 'mlp' or 'rbf'

        # Initialize networks based on type
        if self.network_type == 'mlp':
            self.q_net = Net(self.state_space_dim, 256, self.action_space_dim, self.device).to(self.device)
            self.target_net = Net(self.state_space_dim, 256, self.action_space_dim, self.device).to(self.device)
        else:  # rbf
            self.q_net = RBFNet(self.state_space_dim, self.action_space_dim, self.device).to(self.device)
            self.target_net = RBFNet(self.state_space_dim, self.action_space_dim, self.device).to(self.device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.steps = 0
        self.update_target_steps = 100

    def act(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsi:
            return random.randrange(self.action_space_dim)
        else:
            with torch.no_grad():
                s0 = s0[0] if isinstance(s0, tuple) else s0
                state = torch.tensor(s0, dtype=torch.float).to(self.device)
                q_values = self.q_net(state)
                return torch.argmax(q_values).item()

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample random batch from replay buffer
        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1, done = zip(*samples)
        # s0 = s0[0] if isinstance(s0, tuple) else s0
        # Convert to tensors
        s0 = [s[0] if isinstance(s, tuple) else s for s in s0]
        s1 = [s[0] if isinstance(s, tuple) else s for s in s1]
        s0 = torch.tensor(s0, dtype=torch.float).to(self.device)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1).to(self.device)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(self.device)
        s1 = torch.tensor(s1, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(self.batch_size, -1).to(self.device)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(s1)
            a = torch.max(next_q_values)
            max_next_q = torch.max(next_q_values, dim=1)[0].view(self.batch_size, -1)
            target_q = r1 + (1 - done) * self.gamma * max_next_q

        # Compute current Q values
        current_q = self.q_net(s0).gather(1, a0)

        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.steps % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# Main training loop
def train_dqn(env, agent, num_episodes):
    scores = []
    mean_scores = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated,  _ = env.step(action)
            done = terminated or truncated
            # Store transition in replay buffer
            agent.put(state, action, reward, next_state, done)
            
            # Train the network
            agent.learn()
            
            total_reward += reward
            state = next_state
            
        scores.append(total_reward)
        mean_scores.append(np.mean(scores[-100:]))  # Moving average of last 100 episodes
        
        if episode % 100 == 0:
            print(f'Episode {episode}, Average Score: {mean_scores[-1]:.2f}')
    
    return scores, mean_scores

def compare_networks(env, num_episodes=500, num_runs=5):
    params = {
        'gamma': 0.99,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n,
        'device': device
    }
    
    mlp_results = []
    rbf_results = []
    
    # Run multiple times for statistical significance
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Train MLP
        print("Training MLP Network...")
        params['network_type'] = 'mlp'
        mlp_agent = DQN_Agent(**params)
        mlp_scores, mlp_means = train_dqn(env, mlp_agent, num_episodes)
        mlp_results.append(mlp_means)
        
        # Train RBF
        print("Training RBF Network...")
        params['network_type'] = 'rbf'
        rbf_agent = DQN_Agent(**params)
        rbf_scores, rbf_means = train_dqn(env, rbf_agent, num_episodes)
        rbf_results.append(rbf_means)
    
    # Plot results
    plt.figure(figsize=(15,10))
    
    # Plot MLP results
    mlp_mean = np.mean(mlp_results, axis=0)
    mlp_std = np.std(mlp_results, axis=0)
    plt.plot(mlp_mean, label='MLP Mean', color='blue')
    plt.fill_between(range(len(mlp_mean)), 
                    mlp_mean - mlp_std, 
                    mlp_mean + mlp_std, 
                    alpha=0.2, 
                    color='blue')
    
    # Plot RBF results
    rbf_mean = np.mean(rbf_results, axis=0)
    rbf_std = np.std(rbf_results, axis=0)
    plt.plot(rbf_mean, label='RBF Mean', color='red')
    plt.fill_between(range(len(rbf_mean)), 
                    rbf_mean - rbf_std, 
                    rbf_mean + rbf_std, 
                    alpha=0.2, 
                    color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('MLP vs RBF Network Performance Comparison')
    plt.legend()
    
    # Print final statistics
    print("\nFinal Performance Statistics:")
    print(f"MLP Final Average Score: {mlp_mean[-1]:.2f} ± {mlp_std[-1]:.2f}")
    print(f"RBF Final Average Score: {rbf_mean[-1]:.2f} ± {rbf_std[-1]:.2f}")
    
    plt.show()

# Run comparison
env = gym.make('CartPole-v0')
compare_networks(env, num_episodes=500, num_runs=1)