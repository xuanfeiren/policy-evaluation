
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from stable_baselines3 import PPO

n = n_samples = 10
feature_dim = 3  # Example feature dimension
repeat = 1
gamma = 0.9

env = gym.make('CartPole-v1')
env.reset() #must reset the environment before interacting with it
model_PPO = PPO.load("ppo_cartpole") # PPO policy 

def policy_unif(s):
  a = env.action_space.sample()
  return a

def policy_PPO(s):
  action = model_PPO.predict(s)[0]
  return action

def compute_return(traj,gamma):
  if traj==[]:  
    return 0
  else:
    return traj[2]+gamma*compute_return(traj[3:],gamma)

def rbf_random_fourier_features(state, action, feature_dim = feature_dim, length_scale=1.0):
    np.random.seed(0)
    state_array = np.array(state[0], dtype=np.float32).reshape(-1)
    action_array = np.array([float(action)])
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
        s1, r0, done, _, _ = env.step(a0) # take a random action
        traj_list.append(r0)
        traj_list.append(s1)
        s0 = s1
        if done:
            break
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



def policy_eval_LSTD(theta_init , data,  feature_dim=feature_dim, alpha=0.01):
    '''Use TD(0) which converges to the solution of LSTD'''
    theta = theta_init
    for phi_sa, reward, phi_sa_prime in data:
        Q_sa = np.dot(theta, phi_sa)
        Q_sa_prime = np.dot(theta, phi_sa_prime)
        td_error = reward + gamma * Q_sa_prime - Q_sa
        theta += alpha * td_error * phi_sa
    
    def Q(state, action):
        phi_sa = rbf_random_fourier_features(state, action, feature_dim)
        return np.dot(theta, phi_sa)
    
    return Q

def policy_eval_BRM(data,  feature_dim=feature_dim, alpha=0.01, iterations=100):
    return

theta_init = np.zeros(feature_dim)
offline_data = collect_data(n_samples, policy_PPO, feature_dim)
Q_lstd = policy_eval_LSTD(theta_init, offline_data)
state = env.reset()
action = env.action_space.sample()
estimated_value_lstd_gradient = Q_lstd(state, action)
print(f"Estimated Q-value for the initial state and action using LSTD with gradient updates: {estimated_value_lstd_gradient}")



