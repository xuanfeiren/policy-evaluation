# Mountain-Car-env.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from scipy.optimize import minimize

model_PPO = PPO.load("ppo_mountaincar")

env_name = 'MountainCar-v0'
gym.make(env_name)

n_samples = 10000
feature_dim = 50 # Example feature dimension
repeat = 1
gamma = 0.9
num_grids = 6 # need to be changed in different environment

env = gym.make(env_name)
env.reset()

def policy_unif(s):
  a = env.action_space.sample()
  return a

def policy_PPO(s):
  action = model_PPO.predict(s)[0]
  return action

def rbf_random_fourier_features(state, action, feature_dim = feature_dim, length_scale=1.0):
    if len(state) != 2:
        raise ValueError("State length must be 4")
    np.random.seed(0)
    state_array = np.array(state, dtype=np.float32).reshape(-1)
    action_array = np.array([int(action)])
    state_action = np.concatenate((state_array, action_array))
    dim = state_action.shape[0]
    
    # Handle even/odd feature dimensions
    if feature_dim % 2 == 0:
        d_cos = d_sin = feature_dim // 2
    else:
        d_cos = (feature_dim + 1) // 2
        d_sin = (feature_dim - 1) // 2
    
    omega = np.random.normal(scale=1.0/length_scale, size=(dim, d_cos))
    bias = np.random.uniform(0, 2 * np.pi, size=d_cos)
    z = state_action @ omega + bias
    cos_features = np.cos(z)
    sin_features = np.sin(z[:d_sin]) if d_sin > 0 else np.array([])
    feature = np.sqrt(1.0 / feature_dim) * np.concatenate([cos_features, sin_features])
    return feature

def collect_trajectory(policy, feature_dim):
    s0, _ = env.reset()
    traj_list = [s0]
    while True:
        a0 = policy(s0)
        phi_sa = rbf_random_fourier_features(s0, a0, feature_dim)
        traj_list.append(phi_sa)
        s1, r0,  terminated, truncated, _ = env.step(a0)
        traj_list.append(r0)
        traj_list.append(s1)
        s0 = s1
        if terminated or truncated:
            break
    # print(len(traj_list))
    return traj_list[:-1]  # removing the terminal state

def collect_data(n, policy, feature_dim=feature_dim):
    data = []
    while len(data) < n:
        trajectory = collect_trajectory(policy, feature_dim)
        i = 0
        while i < len(trajectory)-3:
            state = trajectory[i]
            action = policy(state)
            phi_sa = rbf_random_fourier_features(state, action, feature_dim)
            reward = trajectory[i+2]
            next_state = trajectory[i+3]
            next_action = policy(next_state)
            phi_sa_prime = rbf_random_fourier_features(next_state, next_action, feature_dim)
            
            data.append((phi_sa, reward, phi_sa_prime))
            i += 3
            if len(data) >= n:
                break

    return data[:n]  # Return exactly n samples as a single array

def Q(state, action, theta,feature_dim=feature_dim):
    phi_sa = rbf_random_fourier_features(state, action, feature_dim)
    return np.dot(theta, phi_sa)

def policy_eval_LSTD(theta_init,data, feature_dim=feature_dim, alpha=0.01):
    '''Use TD(0) which converges to the solution of LSTD'''
    theta_lstd = np.copy(theta_init)
    for phi_sa, reward, phi_sa_prime in data:
        Q_sa = np.dot(theta_lstd, phi_sa)
        Q_sa_prime = np.dot(theta_lstd, phi_sa_prime)
        td_error = reward + gamma * Q_sa_prime - Q_sa
        theta_lstd += alpha * td_error * phi_sa
    
    # def Q(state, action):
    #     phi_sa = rbf_random_fourier_features(state, action, feature_dim)
    #     return np.dot(theta_lstd, phi_sa)
    
    return theta_lstd

def policy_eval_BRM(theta_init, data,  feature_dim=feature_dim, learning_rate=0.1):
    theta_BRM = np.copy(theta_init)
    for phi_sa, reward, phi_sa_prime in data:
        x_sa = phi_sa - gamma * phi_sa_prime
        gradient = -2 * (reward - np.dot(x_sa, theta_BRM)) * x_sa
        theta_BRM -= learning_rate * gradient
        
    # def Q(state, action):
    #     phi_sa = rbf_random_fourier_features(state, action, feature_dim)
    #     return np.dot(theta_BRM, phi_sa)
    
    return theta_BRM

def index_to_state_action(i, n_grid_points=num_grids):
    """
    Maps index i (0 to 3*n_grid_points^2-1) to a state-action pair
    
    Returns:
    - state: np.array of shape (2,)
    - action: int (0, 1, or 2)
    """
    # State bounds
    state_bounds = [
        [-1.2, 0.6],     # position
        [-0.07, 0.07]    # velocity
    ]
    
    # Total states per dimension
    n_states = n_grid_points**2
    
    # Determine action (0 for first third indices, 1 for second third, 2 for last third)
    action = i // n_states
    
    # Get state index (map back to state space)
    state_idx = i % n_states
    
    # Convert to grid coordinates
    idx_2 = state_idx % n_grid_points
    idx_1 = state_idx // n_grid_points
    
    # Convert grid coordinates to actual state values
    state = np.array([
        np.linspace(state_bounds[0][0], state_bounds[0][1], n_grid_points)[idx_1],
        np.linspace(state_bounds[1][0], state_bounds[1][1], n_grid_points)[idx_2]
    ])
    
    return state, action

def grid_evaluation_pairs(policy, num_grids = num_grids, n_episodes=100, max_steps=200):
    """
    Estimates Q values using given policy for trajectories
    
    Args:
        policy: Function that takes state and returns action
        num_grids: Number of grid points per dimension
        n_episodes: Number of episodes per state-action pair
        max_steps: Maximum steps per episode
        
    Returns:
        Q_vector: Estimated Q-values for each state-action pair
    """
    env = gym.make('MountainCar-v0')
    total_pairs = 3 * num_grids**2 
    Q_vector = np.zeros(total_pairs)
    
    for i in tqdm(range(total_pairs)):
        state, action = index_to_state_action(i, num_grids)
        returns = []
        
        for _ in range(n_episodes):
            env.reset()
            env.state = state
            
            # Take specified initial action
            next_state, reward, term, trunc, _ = env.step(action)
            total_return = reward
            
            # Continue with policy-chosen actions
            discount = gamma
            curr_state = next_state
            steps = 0
            
            while not (term or trunc) and steps < max_steps:
                # Use policy to select action
                curr_action = policy(curr_state)
                curr_state, r, term, trunc, _ = env.step(curr_action)
                total_return += discount * r
                discount *= gamma
                steps += 1
                
                if discount < 1e-10:
                    break
                    
            returns.append(total_return)
            
        Q_vector[i] = np.mean(returns)
    
    env.close()
    return Q_vector

def loss_policy_evaluation(theta, Q_real, num_grids = num_grids):
    loss = 0 
    total_pairs = 3 * num_grids**2 # need to be changed in different environment
    for i in range(total_pairs):
        state, action = index_to_state_action(i, num_grids)
        Q_est_i = Q(state, action, theta)
        loss += (Q_est_i- Q_real[i])**2
    loss /= total_pairs
    # print('Q_est_i:', Q_est_i), print('Q_real:', Q_real[i])
    return loss


Q_real = np.load(f"Q_function_Mountain_car_grid_6.npy")

# Define the objective function for optimization
# def objective_function(theta):
#     return loss_policy_evaluation(theta, Q_real, num_grids)

# # Initial guess for theta_oracle
# theta_init = np.zeros(feature_dim)

# # Perform the optimization to find theta_oracle
# result = minimize(objective_function, theta_init, method='BFGS')

# # Extract the optimized theta_oracle
# theta_oracle = result.x

# Print the optimized theta_oracle and the corresponding loss


iter = int( n_samples / 50 )
loss_LSTD = [0] * int(n_samples/ iter)
loss_BRM = [0] * int(n_samples/ iter)
total_pairs = 3 * num_grids**2 # need to be changed in different environment

for _ in tqdm(range(repeat)):
    l2_norm_diff_BRM_list = []
    l2_norm_diff_LSTD_list = []
    theta_lstd = np.zeros(feature_dim)
    theta_BRM = np.zeros(feature_dim)
    for m in range(iter, n_samples + 1, iter):
        
        offline_data = collect_data(iter, policy_PPO, feature_dim)
        theta_lstd = policy_eval_LSTD(theta_lstd, offline_data)
        theta_BRM = policy_eval_BRM(theta_BRM, offline_data)
        loss_LSTD_m = loss_policy_evaluation(theta_lstd, Q_real)
        loss_BRM_m = loss_policy_evaluation(theta_BRM, Q_real)

        l2_norm_diff_LSTD_list.append(loss_LSTD_m)
        l2_norm_diff_BRM_list.append(loss_BRM_m)
    # print(len(l2_norm_diff_LSTD_list), len(l2_norm_diff_BRM_list))
    loss_LSTD = [a + b for a, b in zip(loss_LSTD, l2_norm_diff_LSTD_list)]
    loss_BRM = [a + b for a, b in zip(loss_BRM, l2_norm_diff_BRM_list)]
loss_LSTD = [value / repeat for value in loss_LSTD]
loss_BRM = [value / repeat for value in loss_BRM]
# loss_oracle = loss_policy_evaluation(theta_oracle, Q_real)

plt.figure(figsize=(10, 6))
# plt.axhline(y=loss_oracle, color='green', linestyle='--', label='Oracle Loss')
plt.plot(range(iter, n_samples + 1, iter), loss_BRM, label='BRM Loss', color='red')
plt.plot(range(iter, n_samples + 1, iter), loss_LSTD, label='LSTD Loss', color='blue')
plt.xlabel('Number of Data Points')
plt.ylabel('L2 Norm Difference')
# plt.yscale('log')
plt.title(f'Loss Curves for BRM and LSTD in {env_name}')
plt.legend()
plt.grid(True)
plt.savefig(f'plot_image_env_{env_name}_n_samples_{n_samples}_feature_dim_{feature_dim}_repeat_{repeat}_gamma_{gamma}_num_grids_{num_grids}.pdf', bbox_inches='tight')
plt.show()