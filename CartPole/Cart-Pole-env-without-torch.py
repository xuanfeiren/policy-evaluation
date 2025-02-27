from importlib.metadata import PathDistribution
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO

from scipy.optimize import minimize


model_PPO = PPO.load("ppo_cartpole")
env_name = 'CartPole-v1'
n_samples = 10000
feature_dim = 100 # Example feature dimension
repeat = 1
gamma = 0.99
num_grids = 3

env = gym.make(env_name)
env.reset()

def policy_unif(s):
  a = env.action_space.sample()
  return a

def policy_PPO(s):
  action = model_PPO.predict(s)[0]
  return action

def policy_mix(mix):
    """
    Creates a policy function with fixed mix parameter
    
    Args:
        mix: Probability of using PPO policy (default 0.75)
    
    Returns:
        Function that takes only state parameter
    """
    def policy(s):
        if np.random.rand() < mix:
            return policy_PPO(s)
        else:
            return policy_unif(s)
    return policy

def fourier_features(state, action, feature_dim = feature_dim, length_scale=1.0):
    # np.random.seed(0)
    state_array = np.array(state, dtype=np.float32).reshape(-1)
    action_array = np.array([int(action)])
    state_action = np.concatenate((state_array, action_array))
    dim = state_action.shape[0]
    omega = np.zeros((dim, feature_dim))
    
    for j in range(feature_dim):
        # Create frequency multipliers that increase with column index
        freq = (j + 1) * np.pi
        # Fill column with increasing frequencies for each dimension
        for i in range(dim):
            omega[i,j] = freq * (i + 1)
    feature =  np.cos( state_action @ omega)
    return feature

# seed = np.random.randint(0, 10000)
# print(f"Random seed: {seed}")
def RFFeatures(state, action, feature_dim = feature_dim, length_scale=1,seed=0):
    # return fourier_features(state, action, feature_dim, length_scale)
    if len(state) != 4:
        raise ValueError("State length must be 4")
    np.random.seed(seed)
    state_array = np.array(state, dtype=np.float32).reshape(-1)
    action_array = np.array([int(action)])
    state_action = np.concatenate((state_array, action_array))
    # Normalize state_action to [0, 1]
    # state_action = (state_action - state_action.min()) / (state_action.max() - state_action.min())
    # return state_action
    dim = state_action.shape[0]

    feature_dim = feature_dim - 1

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

    feature = np.concatenate((feature,np.sqrt(1.0 / feature_dim) *action_array))
    return feature

def collect_trajectory(policy, feature_dim):
    s0, _ = env.reset()
    traj_list = [s0]
    while True:
        a0 = policy(s0)
        phi_sa = linear_features(s0, a0, feature_dim)
        traj_list.append(phi_sa)
        s1, r0,  terminated, truncated, _ = env.step(a0)
        traj_list.append(r0)
        traj_list.append(s1)
        s0 = s1
        if terminated or truncated:
            break
    # print(len(traj_list))
    return traj_list

def collect_data(n,policy_to_gen_data, policy_to_eval, feature_dim=feature_dim):
    data = []
    while len(data) < n:
        trajectory = collect_trajectory(policy_to_gen_data, feature_dim)
        i = 0
        while i <= len(trajectory)-3:
            state = trajectory[i]
            # action = policy(state)
            phi_sa = trajectory[i+1]
            reward = trajectory[i+2]
            next_state = trajectory[i+3]
            next_action = policy_to_eval(next_state)
            phi_sa_prime = linear_features(next_state, next_action, feature_dim)
            
            data.append((phi_sa, reward, phi_sa_prime))
            i += 3
            if len(data) >= n:
                break

    return data[:n]  # Return exactly n samples as a single array

def Q(state, action, theta,feature_dim=feature_dim):
    phi_sa = linear_features(state, action, feature_dim)
    return np.dot(theta, phi_sa)

def policy_eval_LSTD(theta_init,data, feature_dim=feature_dim, alpha=1):
    '''Use TD(0) which converges to the solution of LSTD'''
    theta_lstd = np.copy(theta_init)
    for phi_sa, reward, phi_sa_prime in data:
        Q_sa = np.dot(theta_lstd, phi_sa)
        Q_sa_prime = np.dot(theta_lstd, phi_sa_prime)
        td_error = reward + gamma * Q_sa_prime - Q_sa
        theta_lstd += alpha * td_error * phi_sa
    
    # def Q(state, action):
    #     phi_sa = linear_features(state, action, feature_dim)
    #     return np.dot(theta_lstd, phi_sa)
    
    return theta_lstd

def policy_eval_FQI(theta_init,data, num_FQI=10):
    theta_FQI = np.copy(theta_init)
    m = len(data)
    dim = theta_init.shape[0]
    Sigma_cov = np.zeros((dim, dim))
    theta_phi_r = np.zeros(dim)
    Sigma_cr = np.zeros((dim, dim))

    for phi_sa, reward, phi_sa_prime in data:
        Sigma_cov += np.outer(phi_sa, phi_sa)
        theta_phi_r += reward * phi_sa
        Sigma_cr += np.outer(phi_sa, phi_sa_prime)
    
    Sigma_cov /= m
    theta_phi_r /= m
    Sigma_cr /= m
    for _ in range(num_FQI):
            # use m data to update theta
            theta_FQI = np.linalg.inv( Sigma_cov) @ ( theta_phi_r+  gamma * Sigma_cr @ theta_FQI)
    return theta_FQI

def policy_eval_BRM(theta_init, data,  feature_dim=feature_dim, learning_rate=1):
    theta_BRM = np.copy(theta_init)
    for phi_sa, reward, phi_sa_prime in data:
        x_sa = phi_sa - gamma * phi_sa_prime
        gradient = 2 * (np.dot(x_sa, theta_BRM) - reward) * x_sa
        theta_BRM -= learning_rate * gradient
        
    # def Q(state, action):
    #     phi_sa = linear_features(state, action, feature_dim)
    #     return np.dot(theta_BRM, phi_sa)
    
    return theta_BRM

def index_to_state_action(i, n_grid_points=num_grids):
    """
    Maps index i (0 to 2*n_grid_points^4-1) to a state-action pair
    
    Returns:
    - state: np.array of shape (4,)
    - action: int (0 or 1)
    """
    # State bounds
    state_bounds = [
        [-4.8, 4.8],     # cart position
        [-10.0, 10.0],   # cart velocity
        [-0.418, 0.418], # pole angle
        [-10.0, 10.0]    # pole angular velocity
    ]
    
    # Total states per dimension
    n_states = n_grid_points**4
    
    # Determine action (0 for first half indices, 1 for second half)
    action = 1 if i>= n_states else 0
    
    # Get state index (map back to state space)
    state_idx = i % n_states
    
    # Convert to grid coordinates
    idx_4 = state_idx % n_grid_points
    idx_3 = (state_idx // n_grid_points) % n_grid_points
    idx_2 = (state_idx // (n_grid_points**2)) % n_grid_points
    idx_1 = state_idx // (n_grid_points**3)
    
    # Convert grid coordinates to actual state values
    state = np.array([
        np.linspace(state_bounds[0][0], state_bounds[0][1], n_grid_points)[idx_1],
        np.linspace(state_bounds[1][0], state_bounds[1][1], n_grid_points)[idx_2],
        np.linspace(state_bounds[2][0], state_bounds[2][1], n_grid_points)[idx_3],
        np.linspace(state_bounds[3][0], state_bounds[3][1], n_grid_points)[idx_4]
    ])
    
    return state, action

def grid_evaluation_pairs(policy, num_grids = num_grids, n_episodes=100, max_steps=500):
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
    env = gym.make('CartPole-v1')
    total_pairs = 2 * num_grids**4
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
    total_pairs = 2 * num_grids**4
    for i in range(total_pairs):
        state, action = index_to_state_action(i, num_grids)
        Q_est_i = Q(state, action, theta)
        # Q_real[i] = 0
        loss += (Q_est_i- Q_real[i])**2
        # print(f"Q_est: {Q_est_i}, Q_real: {Q_real[i]}")
    loss /= total_pairs
    return loss

# def more_loss_policy_evaluation(theta,  num_grids ):
#     loss = 0 
#     total_pairs = 2 * num_grids**4
#     dim = theta.shape[0]
#     for i in range(total_pairs):
#         state, action = index_to_state_action(i, num_grids)
#         Q_est_i = Q(state, action, theta,feature_dim = dim)
#         loss += (Q_est_i- 10)**2
#         # print(f"Q_est: {Q_est_i}, Q_real: {Q_real[i]}")
#     loss /= total_pairs
#     return loss

# def find_optimal_theta(feature_dim, num_grids=4):
#     theta_init = np.zeros(feature_dim)
#     pbar = tqdm(total=105000, desc='Optimizing theta')
#     n_evals = 0
    
#     def objective_with_progress(theta):
#         nonlocal n_evals
#         n_evals += 1
#         pbar.update(1)
#         return more_loss_policy_evaluation(theta, num_grids)
    
#     try:
#         result = minimize(
#             fun=objective_with_progress,
#             x0=theta_init,
#             method='BFGS',
#             options={'maxiter': 1000}
#         )
        
#         print(f"\nFinal loss: {result.fun:.6f}")
#         print(f"BFGS iterations: {result.nit}")
#         print(f"Function evaluations: {n_evals}")
#         print(f"Average evaluations per iteration: {n_evals/result.nit:.1f}")
        
#     finally:
#         pbar.close()
    
#     return result.x
# def get_optimal_theta_ridge(feature_dim, num_grids = 20, lambda_reg=0.1):
#     """
#     Find optimal theta using ridge regression
#     """
#     total_pairs = 2 * num_grids**4
    
#     # Build feature matrix Φ
#     Phi = np.zeros((total_pairs, feature_dim))
#     for i in range(total_pairs):
#         state, action = index_to_state_action(i, num_grids)
#         Phi[i] = linear_features(state, action, feature_dim)
    
#     # Target vector
#     y = 10 * np.ones(total_pairs)
    
#     # Ridge regression solution: θ* = (Φ^T Φ + λI)^(-1) Φ^T y
#     I = np.eye(feature_dim)
#     theta_ridge = np.linalg.inv(Phi.T @ Phi + lambda_reg * I) @ Phi.T @ y
    
#     # Calculate minimized loss including regularization term
#     pred = Phi @ theta_ridge
#     mse_loss = np.mean((pred - y)**2)
    
    
#     print(f"MSE loss: {mse_loss:.6f}")
#     loss_eval = more_loss_policy_evaluation(theta_ridge, 3 )
#     print( f"Loss evaluation: {loss_eval:.6f}")

#     return theta_ridge
# get_optimal_theta_ridge(1000)

Q_real = np.load(f"Q_function_grid_3_gamma_0.99.npy")
seed_to_traverse = 0
def linear_features(state, action, feature_dim):
    return RFFeatures(state, action, feature_dim, seed=seed_to_traverse)
    
iter = int( n_samples / 50 )
loss_LSTD = [0] * int(n_samples/ iter)
loss_BRM = [0] * int(n_samples/ iter)
total_pairs = 2 * num_grids**4 

for _ in (range(repeat)):
    l2_norm_diff_BRM_list = []
    l2_norm_diff_LSTD_list = []
    theta_lstd = np.zeros(feature_dim)
    # theta_lstd = np.random.normal(0, 1, feature_dim)
    theta_BRM = np.zeros(feature_dim)
    for m in range(iter, n_samples + 1, iter):
        
        offline_data = collect_data(iter, policy_mix(1),policy_PPO, feature_dim)
        theta_lstd = policy_eval_LSTD(theta_lstd, offline_data)
        theta_BRM = policy_eval_BRM(theta_BRM, offline_data)
        loss_LSTD_m = loss_policy_evaluation(theta_lstd, Q_real)
        loss_BRM_m = loss_policy_evaluation(theta_BRM, Q_real)

        l2_norm_diff_LSTD_list.append(loss_LSTD_m)
        l2_norm_diff_BRM_list.append(loss_BRM_m)
        # print(f"Loss LSTD: {loss_LSTD_m}, Loss BRM: {loss_BRM_m}")
    # print(len(l2_norm_diff_LSTD_list), len(l2_norm_diff_BRM_list))
    loss_LSTD = [a + b for a, b in zip(loss_LSTD, l2_norm_diff_LSTD_list)]
    loss_BRM = [a + b for a, b in zip(loss_BRM, l2_norm_diff_BRM_list)]
loss_LSTD = [value / repeat for value in loss_LSTD]
loss_BRM = [value / repeat for value in loss_BRM]

# theta_FQI = np.zeros(feature_dim)
# offline_data = collect_data(n_samples, policy_PPO,policy_PPO, feature_dim)
# theta_FQI = policy_eval_FQI(theta_FQI, offline_data)
# loss_FQI = loss_policy_evaluation(theta_FQI, Q_real)
# print(f"Loss FQI: {loss_FQI}")




plt.figure(figsize=(10, 6))
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
        



