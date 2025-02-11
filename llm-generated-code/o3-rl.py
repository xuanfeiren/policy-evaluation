# openai-o3-mini-high
#!/usr/bin/env python3
"""
Q-learning with linear function approximation using RBF features
for Gymnasium classic control environments (Acrobot, CartPole, MountainCar, Pendulum).

Usage:
    python qlearning_rbf.py --env CartPole-v1 --episodes 500 --render

To switch environments, simply change the --env argument (e.g., "MountainCar-v0", "Acrobot-v1", "Pendulum-v1").
Note: For continuous-action environments (e.g., Pendulum-v1), actions are discretized.
"""

import argparse
import gymnasium as gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

# =============================
# Feature Transformer Functions
# =============================

def create_feature_transformer(env, n_samples=10000):
    """
    Create a feature transformer that uses RBF kernels.
    First, sample many states from the environment (using random actions)
    to fit a StandardScaler and then a FeatureUnion of multiple RBFSamplers
    with different gamma values.
    """
    observations = []
    state, _ = env.reset()
    for _ in range(n_samples):
        # Take a random action
        action = env.action_space.sample()
        next_state, _, done, truncated, _ = env.step(action)
        observations.append(state)
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    observations = np.array(observations)
    
    # Standardize the observations so that each feature has mean=0 and unit variance.
    scaler = StandardScaler()
    scaler.fit(observations)
    
    # Create a feature union of RBF kernels with different gamma values
    featurizer = FeatureUnion([
        ("rbf1", RBFSampler(gamma=0.05, n_components=100, random_state=123)),
        ("rbf2", RBFSampler(gamma=0.1,  n_components=100, random_state=123)),
        ("rbf3", RBFSampler(gamma=0.5,  n_components=100, random_state=123)),
        ("rbf4", RBFSampler(gamma=1.0,  n_components=100, random_state=123))
    ])
    # Fit the featurizer on the scaled observations.
    featurizer.fit(scaler.transform(observations))
    
    return scaler, featurizer

def featurize_state(state, scaler, featurizer):
    """
    Given a state, first scale it and then transform it using the RBF featurizer.
    Returns a 1D feature vector.
    """
    scaled = scaler.transform([state])
    features = featurizer.transform(scaled)
    return features[0]

# =============================
# Q-Learning Agent Definition
# =============================

class QLearner:
    """
    Q-learning agent with linear function approximation.
    The Q-value for a given state and action is approximated as the dot product:
        Q(s, a) = w_a^T * phi(s)
    where phi(s) are the features of the state.
    """
    def __init__(self, n_actions, feature_dim, alpha=0.01, gamma=0.99):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        # Weight matrix: one row per action, each of length feature_dim.
        self.weights = np.zeros((n_actions, feature_dim))
        self.alpha = alpha
        self.gamma = gamma

    def predict(self, features):
        """
        Compute Q-values for all actions given state features.
        """
        return self.weights.dot(features)

    def update(self, features, action, target):
        """
        Update the weight vector for the taken action using the TD error.
        """
        prediction = self.predict(features)[action]
        error = target - prediction
        self.weights[action] += self.alpha * error * features

    def select_action(self, features, epsilon):
        """
        Select an action using epsilon-greedy policy.
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_values = self.predict(features)
            return np.argmax(q_values)

# =============================
# Action Space Discretization
# =============================

def discretize_action_space(env, num_bins=11):
    """
    For continuous action spaces (assumed to be 1-dimensional), discretize the actions.
    For example, in Pendulum-v1 the action (torque) is continuous between low and high.
    """
    low = env.action_space.low[0]
    high = env.action_space.high[0]
    discrete_actions = np.linspace(low, high, num_bins)
    return discrete_actions

# =============================
# Main Training Loop
# =============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Name of the Gymnasium environment (e.g., CartPole-v1, MountainCar-v0, Acrobot-v1, Pendulum-v1)')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--alpha', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for epsilon-greedy policy')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()

    # Create the environment
    env = gym.make(args.env)
    # Check whether the action space is continuous (Box) or discrete.
    is_continuous_action = isinstance(env.action_space, gym.spaces.Box)
    if is_continuous_action:
        discrete_actions = discretize_action_space(env)
        n_actions = len(discrete_actions)
    else:
        n_actions = env.action_space.n

    # Create feature transformer using samples from the environment.
    print("Collecting samples to fit the feature transformer...")
    scaler, featurizer = create_feature_transformer(env)
    # Get the dimensionality of the transformed feature space.
    sample_state, _ = env.reset()
    features_sample = featurize_state(sample_state, scaler, featurizer)
    feature_dim = features_sample.shape[0]
    print(f"Feature dimension: {feature_dim}")

    # Instantiate the Q-learning agent.
    agent = QLearner(n_actions, feature_dim, alpha=args.alpha, gamma=args.gamma)

    # Training loop.
    rewards = []
    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            if args.render:
                env.render()
            features = featurize_state(state, scaler, featurizer)
            action_idx = agent.select_action(features, args.epsilon)
            # For continuous action spaces, select the discretized action.
            if is_continuous_action:
                action = np.array([discrete_actions[action_idx]])
            else:
                action = action_idx
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            next_features = featurize_state(next_state, scaler, featurizer)
            # Compute the Q-learning target.
            q_next = agent.predict(next_features)
            target = reward
            if not (done or truncated):
                target += args.gamma * np.max(q_next)
            # Update the agent.
            agent.update(features, action_idx, target)
            state = next_state
        rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{args.episodes}: Reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    main()
