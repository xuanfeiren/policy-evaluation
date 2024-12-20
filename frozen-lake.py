# frozen-lake.py
import re
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



n_samples = 1000
feature_dim = 1 # Example feature dimension
repeat = 1
gamma = 0.99
env_name = 'FrozenLake-v1'
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

env.reset()

def policy_unif(s):
  a = env.action_space.sample()
  return a

def best_policy(s):
  policy_map = {
    0: 1, 1: 2, 2: 1, 3: 0,
    4: 1, 5: 1, 6: 1, 7: 2,
    8: 2, 9: 2, 10: 1, 11: 2,
    12: 1, 13: 2, 14: 2, 15: 2
  }
  return policy_map.get(s, env.action_space.sample())

def best_policy_rand(s):
  random_value = np.random.rand()
  if random_value < 0.95:
    return best_policy(s)
  else:
    return policy_unif(s)

def rbf_random_fourier_features(state, action, feature_dim = feature_dim, length_scale=1.0):
    # return fourier_features(state, action, feature_dim)
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
    return traj_list  # removing the terminal state

def collect_data(n, policy_to_gen_data, policy_to_eval, feature_dim=feature_dim):
    data = []
    while len(data) < n:
        trajectory = collect_trajectory(policy_to_gen_data, feature_dim)
        i = 0
        while i <= len(trajectory)-3:
            state = trajectory[i]
            phi_sa = trajectory[i+1]
            reward = trajectory[i+2]
            print(reward)
            next_state = trajectory[i+3]
            next_action = policy_to_eval(next_state)
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

n_states = env.observation_space.n
n_actions = env.action_space.n

def compute_return(traj):
  if traj==[]:
    return 0
  else:
    return traj[2]+gamma*compute_return(traj[3:])
  
def collect_trajectory_s_a(policy,s0,a0):
  env.reset()
  env.unwrapped.s = s0
  traj_list = [s0,a0]
  s1, r0, terminated, truncated, _ = env.step(a0)
  traj_list.append(r0)
  traj_list.append(s1)
  s0=s1
  if terminated or truncated:
    return traj_list[:-1]
  
  while True:
    a0 = policy(s0)
    traj_list.append(a0)
    s1, r0, terminated, truncated, _ = env.step(a0)
    traj_list.append(r0)
    traj_list.append(s1)
    s0 = s1
    if terminated or truncated:
      break
  return traj_list[:-1] #removing the terminal state

def compute_Q_real(policy):
    Q = np.zeros((n_states, n_actions))
    for state in range(n_states):
        for action in range(n_actions):
            traj = collect_trajectory_s_a(policy, state, action)
            Q[state, action] = compute_return(traj)
    return Q

# Calculate Q_real for the best policy
Q_real = compute_Q_real(best_policy)

def loss_policy_evaluation(theta, Q_real):
    loss = 0 
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    for state in range(n_states):
        for action in range(n_actions):
            Q_est_i = Q(state, action, theta)
            loss += (Q_est_i- Q_real[state,action])**2
    loss /= n_states * n_actions
    return loss


iter = int( n_samples /10 )
loss_LSTD = [0] * int(n_samples/ iter)
loss_BRM = [0] * int(n_samples/ iter)

for _ in tqdm(range(repeat)):
    l2_norm_diff_BRM_list = []
    l2_norm_diff_LSTD_list = []
    theta_lstd = np.zeros(feature_dim)
    theta_BRM = np.zeros(feature_dim)
    for m in range(iter, n_samples + 1, iter):
        
        offline_data = collect_data(iter, best_policy_rand,best_policy, feature_dim)
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
# plt.savefig(f'plot_image_env_{env_name}_n_samples_{n_samples}_feature_dim_{feature_dim}_repeat_{repeat}_gamma_{gamma}.pdf', bbox_inches='tight')
plt.show()