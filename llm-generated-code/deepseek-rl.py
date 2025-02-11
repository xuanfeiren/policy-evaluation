# deepseek-rl
import gymnasium as gym
import numpy as np

# Configuration - Modify these parameters as needed
# ENV_NAME = "CartPole-v1"        # Change this to switch environments
# ENV_NAME = "Pendulum-v1"        # Change this to switch environments
ENV_NAME = "MountainCar-v0"      # Change this to switch environments

# ENV_NAME = "Acrobot-v1"          # Change this to switch environments
N_FEATURES = 200                # Number of random Fourier features
RBF_BANDWIDTH = 1.0             # Bandwidth for RBF kernel approximation
ALPHA = 0.01                    # Learning rate
GAMMA = 0.99                    # Discount factor
EPSILON = 0.1                   # Exploration rate
NUM_EPISODES = 30000          # Training episodes
NUM_DISCRETE_ACTIONS = 5        # For continuous action spaces

def main():
    # Create environment
    env = gym.make(ENV_NAME)
    if ENV_NAME == "MountainCar-v0":
        env.spec.max_episode_steps = 1000

    # Discretize action space if necessary
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = env.action_space.n
        actions = list(range(num_actions))
    else:  # Handle continuous action space
        assert env.action_space.shape[0] == 1, "Only 1D continuous actions supported"
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]
        actions = np.linspace(action_low, action_high, NUM_DISCRETE_ACTIONS)
        actions = [np.array([a]) for a in actions]
        num_actions = NUM_DISCRETE_ACTIONS
    
    # Get state space dimensions
    state_dim = env.observation_space.shape[0]
    
    # Initialize Random Fourier Features
    np.random.seed(0)  # For reproducibility
    W = np.random.normal(scale=RBF_BANDWIDTH, size=(N_FEATURES, state_dim))
    b = np.random.uniform(0, 2*np.pi, size=N_FEATURES)
    feature_scale = np.sqrt(2.0 / N_FEATURES)
    
    # Initialize weights matrix (actions x features)
    weights = np.zeros((num_actions, N_FEATURES))
    
    # Feature extraction function
    def get_features(state):
        projection = W @ state + b
        return feature_scale * np.cos(projection)
    
    # Training loop
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Compute features and Q-values
            phi = get_features(state)
            q_values = weights @ phi  # Direct matrix-vector multiplication
            
            # Epsilon-greedy action selection
            if np.random.rand() < EPSILON:
                action_idx = np.random.randint(num_actions)
            else:
                action_idx = np.argmax(q_values)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(actions[action_idx])
            done = terminated or truncated
            
            # Calculate TD target
            if done:
                target = reward
            else:
                next_phi = get_features(next_state)
                target = reward + GAMMA * np.max(weights @ next_phi)
            
            # Update weights
            td_error = target - q_values[action_idx]
            weights[action_idx] += ALPHA * td_error * phi
            
            total_reward += reward
            state = next_state
        
        print(f"Episode {episode+1:4d}, Total Reward: {total_reward:6.1f}")
    
    env.close()

if __name__ == "__main__":
    main()