# deepseek-rl-policy-evaluation.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plte

# Configuration
ENV_NAME = "CartPole-v1"
N_FEATURES = 200
RBF_BANDWIDTH = 1.0
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1
NUM_EPISODES = 300
EVAL_INTERVAL = 20
NUM_DISCRETE_ACTIONS = 1  # Not used for CartPole
MC_ROLLOUTS = 5
BUFFER_SIZE = 10000
NUM_EVAL_PAIRS = 50

class QLearner:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        
        # Initialize RFF
        self.W = np.random.normal(scale=RBF_BANDWIDTH, size=(N_FEATURES, self.state_dim))
        self.b = np.random.uniform(0, 2*np.pi, size=N_FEATURES)
        self.feature_scale = np.sqrt(2.0 / N_FEATURES)
        
        # Initialize weights
        self.num_actions = env.action_space.n
        self.weights = np.zeros((self.num_actions, N_FEATURES))
        
        # Replay buffer
        self.buffer = []
        
    def get_features(self, state):
        projection = self.W @ state + self.b
        return self.feature_scale * np.cos(projection)
    
    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        phi = self.get_features(state)
        q_values = self.weights @ phi
        return np.argmax(q_values)
    
    def train(self, num_episodes):
        losses = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.get_action(state, EPSILON)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.buffer.append((state, action, reward, next_state, done))
                if len(self.buffer) > BUFFER_SIZE:
                    self.buffer.pop(0)
                
                # Q-learning update
                phi = self.get_features(state)
                next_phi = self.get_features(next_state)
                
                current_q = self.weights[action] @ phi
                next_q = np.max(self.weights @ next_phi) if not done else 0
                td_error = reward + GAMMA * next_q - current_q
                self.weights[action] += ALPHA * td_error * phi
                
                total_reward += reward
                state = next_state
            
            # Periodic evaluation
            if episode % EVAL_INTERVAL == 0:
                mse = self.evaluate_policy()
                losses.append((episode, mse))
                print(f"Episode {episode:4d}, MSE: {mse:.4f}")
        
        return losses
    
    def mc_return(self, state, action):
        total_return = 0
        for _ in range(MC_ROLLOUTS):
            s, _ = self.env.reset(seed=np.random.randint(0, 10000))
            self.env.unwrapped.state = state
            done = False
            discount = 1.0
            current_return = 0
            
            # Take the specified initial action
            s, r, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            current_return += r * discount
            discount *= GAMMA
            
            while not done:
                a = self.get_action(s, epsilon=0)  # Greedy policy
                s, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                current_return += r * discount
                discount *= GAMMA
            
            total_return += current_return / MC_ROLLOUTS
        return total_return
    
    def evaluate_policy(self):
        if len(self.buffer) < 100:
            return 0
        
        # Randomly select state-action pairs
        indices = np.random.choice(len(self.buffer), size=NUM_EVAL_PAIRS)
        eval_pairs = [self.buffer[i] for i in indices]
        
        # Calculate MC returns
        mc_returns = []
        for state, action, _, _, _ in eval_pairs:
            mc_returns.append(self.mc_return(state, action))
        
        # Calculate LSTD estimates
        lstd_q = self.lstd_evaluation(eval_pairs)
        
        # Calculate MSE
        return np.mean((np.array(mc_returns) - np.array(lstd_q))**2)
    
    def lstd_evaluation(self, eval_pairs):
        # Build LSTD matrices
        A = np.zeros((N_FEATURES*self.num_actions, N_FEATURES*self.num_actions))
        b = np.zeros(N_FEATURES*self.num_actions)
        
        for state, action, reward, next_state, done in self.buffer:
            phi = self.get_features(state)
            next_phi = self.get_features(next_state)
            next_action = np.argmax(self.weights @ next_phi)
            
            # Create feature vectors
            x = np.zeros(N_FEATURES*self.num_actions)
            start = action * N_FEATURES
            x[start:start+N_FEATURES] = phi
            
            x_prime = np.zeros(N_FEATURES*self.num_actions)
            if not done:
                start_prime = next_action * N_FEATURES
                x_prime[start_prime:start_prime+N_FEATURES] = next_phi
            
            # Update matrices
            A += np.outer(x, x - GAMMA*x_prime)
            b += x * reward
        
        # Solve LSTD
        theta = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Calculate Q-values for evaluation pairs
        q_values = []
        for state, action, _, _, _ in eval_pairs:
            phi = self.get_features(state)
            start = action * N_FEATURES
            q = theta[start:start+N_FEATURES] @ phi
            q_values.append(q)
        
        return q_values

def main():
    env = gym.make(ENV_NAME)
    q_learner = QLearner(env)
    losses = q_learner.train(NUM_EPISODES)
    env.close()
    
    # Plot results
    episodes, mses = zip(*losses)
    plt.plot(episodes, mses)
    plt.xlabel('Training Episodes')
    plt.ylabel('MSE (LSTD vs MC)')
    plt.title('Policy Evaluation Error During Training')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()