import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
# Helper function to calculate operator norm (maximum singular value)

def operator_norm(matrix):
    return np.linalg.norm(matrix, ord=2)  # 2-norm is the operator norm

def mu_operator_norm(matrix, D):
    # Calculate D^(1/2) and D^(-1/2) using scipy's matrix square root
    D_sqrt = sqrtm(D)
    D_neg_sqrt = sqrtm(np.linalg.inv(D))
    
    # Calculate weighted norm
    matrix = D_sqrt @ matrix @ D_neg_sqrt
    return np.linalg.norm(matrix, ord=2)
# Parameters

s = 5  # Size of P and Phi
feature_dim = 2 # Feature dimension
gamma = 0.95 # discounting factor

# Set random seed for reproducibility
seed = np.random.randint(0, 10000)
# seed = 2153 # BRM has larger bound on this seed
np.random.seed(seed)
# print(f"Random seed: {seed}")

# Data distribution matrix
# Create diagonal data distribution matrix D with probabilities summing to 1
d = np.random.random(s)
d = d / np.sum(d)  # Normalize to sum to 1
D = np.diag(d)

# Randomly generate P as a s*s stochastic matrix with positive entries (each row sums to 1)
P = np.random.random((s, s))
P = P / P.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1

# Generate a deterministic transition matrix P
P = np.zeros((s, s))
for i in range(s):
    next_state = np.random.choice(s)
    P[i, next_state] = 1
# Random generate Phi as a sxd matrix
Phi = np.random.uniform(-10, 10, size=(s, feature_dim))
# Compute M = I - gamma * P
I = np.eye(s)
M = I - gamma * P
# Compute v = (I - gamma * P)^(-1) * R

theta_real = np.random.uniform(-10, 10, size=(feature_dim, 1))
v = Phi @ theta_real 
v += 1e-3 * np.random.randn(*v.shape)  # Add small noise to v
R = v - gamma * (P @ v)
# v = np.linalg.inv(I - gamma * P) @ R
# print("Real value function v:", v)
# Number of data points

# Find theta_star to minimize ||Phi * theta - v||_2
theta_star = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ v
v_star = Phi @ theta_star

n = 10000
# Generate data
data = []
for _ in range(n):
    # Sample state s_i from distribution D
    s_i = np.random.choice(s, p=d)
    
    # Sample next state s_i' given s_i from transition matrix P
    s_i_prime = np.random.choice(s, p=P[s_i])
    
    # Get feature vector and reward for state s_i
    phi_s_i = Phi[s_i]
    r_s_i = R[s_i]
    
    # Get feature vector for next state s_i'
    phi_s_i_prime = Phi[s_i_prime]
    
    # Append to data
    data.append((phi_s_i, r_s_i, phi_s_i_prime))

# Convert data to numpy arrays for easier manipulation
phi_s = np.array([d[0] for d in data])
r_s = np.array([d[1] for d in data])
phi_s_prime = np.array([d[2] for d in data])
D

# Initialize lists to store L2 norm differences for BRM and LSTD
l2_norm_diff_BRM_list = []
l2_norm_diff_LSTD_list = []

# Calculate real covariance matrices using the real distribution d and transition matrix P
Sigma_cov_real = np.zeros((feature_dim, feature_dim))
for i in range(s):
    Sigma_cov_real += d[i] * np.outer(Phi[i], Phi[i])

Sigma_cr_real = np.zeros((feature_dim, feature_dim))
for i in range(s):
    for j in range(s):
        Sigma_cr_real += d[i] * P[i, j] * np.outer(Phi[i], Phi[j])

Sigma_next_real = np.zeros((feature_dim, feature_dim))
for i in range(s):
    for j in range(s):
        Sigma_next_real += d[i] * P[i,j] * np.outer(Phi[j], Phi[j])

theta_phi_r_real = np.zeros((feature_dim, 1))
for i in range(s):
    theta_phi_r_real += d[i] * (Phi[i][:, np.newaxis] * R[i])

theta_phi_prime_r_real = np.zeros((feature_dim, 1))
for i in range(s):
    for j in range(s):
        theta_phi_prime_r_real += d[i] * P[i, j] * (Phi[j][:, np.newaxis] * R[i])
iter = int(n/50)
# Loop over different sizes of data from 100 to 10000 in steps of 100
for m in range(iter, n + 1, iter):
    # Use first m data points
    phi_s_m = phi_s[:m]
    r_s_m = r_s[:m]
    phi_s_prime_m = phi_s_prime[:m]
    
    # Calculate covariance matrices for first m data points
    Sigma_cov_m = np.zeros((feature_dim, feature_dim))
    for phi in phi_s_m:
        Sigma_cov_m += np.outer(phi, phi)
    Sigma_cov_m /= m
    
    Sigma_cr_m = np.zeros((feature_dim, feature_dim))
    for phi, phi_prime in zip(phi_s_m, phi_s_prime_m):
        Sigma_cr_m += np.outer(phi, phi_prime)
    Sigma_cr_m /= m
    
    Sigma_next_m = np.zeros((feature_dim, feature_dim))
    for phi_prime in phi_s_prime_m:
        Sigma_next_m += np.outer(phi_prime, phi_prime)
    Sigma_next_m /= m
    
    theta_phi_r_m = np.zeros((feature_dim, 1))
    for phi, r in zip(phi_s_m, r_s_m):
        theta_phi_r_m += (phi[:, np.newaxis] * r)
    theta_phi_r_m /= m
    
    theta_phi_prime_r_m = np.zeros((feature_dim, 1))
    for phi_prime, r in zip(phi_s_prime_m, r_s_m):
        theta_phi_prime_r_m += (phi_prime[:, np.newaxis] * r)
    theta_phi_prime_r_m /= m
    
    # BRM estimator for first m data points
    theta_hat_BRM_m = np.linalg.inv(Sigma_cov_m - gamma * Sigma_cr_m - gamma * Sigma_cr_m.T + gamma**2 * Sigma_next_m) @ (theta_phi_r_m - gamma * theta_phi_prime_r_m)
    v_hat_BRM_m = Phi @ theta_hat_BRM_m
    

    l2_norm_diff_BRM_m = np.linalg.norm(v - v_hat_BRM_m, ord=2)

    l2_norm_diff_BRM_list.append(l2_norm_diff_BRM_m)
    
    # LSTD estimator for first m data points
    theta_hat_LSTD_m = np.linalg.inv(Sigma_cov_m - gamma * Sigma_cr_m) @ (theta_phi_r_m)
    v_hat_LSTD_m = Phi @ theta_hat_LSTD_m
    l2_norm_diff_LSTD_m = np.linalg.norm(v - v_hat_LSTD_m, ord=2)
    l2_norm_diff_LSTD_list.append(l2_norm_diff_LSTD_m)

# estimator with infinite data points
theta_hat_BRM_real = np.linalg.inv(Sigma_cov_real - gamma * Sigma_cr_real - gamma * Sigma_cr_real.T + gamma**2 * Sigma_next_real) @ (theta_phi_r_real - gamma * theta_phi_prime_r_real)
v_har_BRM_real = Phi @ theta_hat_BRM_real
Loss_BRM_real = np.linalg.norm(v - v_har_BRM_real, ord=2)

theta_hat_LSTD_real = np.linalg.inv(Sigma_cov_real - gamma * Sigma_cr_real) @ (theta_phi_r_real)
v_hat_LSTD_real = Phi @ theta_hat_LSTD_real
Loss_LSTD_real = np.linalg.norm(v - v_hat_LSTD_real, ord=2)

Loss_oracle = Loss_v_star = np.linalg.norm(v - v_star, ord=2)
alpha_LSTD = Loss_LSTD_real/Loss_oracle
alpha_BRM = Loss_BRM_real/Loss_oracle
# Plot loss curves
plt.figure(figsize=(10, 6))
plt.text(n, Loss_LSTD_real, f'alpha_LSTD = {alpha_LSTD:.2f}', color='blue', verticalalignment='bottom')
plt.text(n, Loss_BRM_real, f'alpha_BRM = {alpha_BRM:.2f}', color='red', verticalalignment='bottom')
plt.plot(range(iter, n + 1, iter), l2_norm_diff_BRM_list, label='BRM Loss', color='red')
plt.plot(range(iter, n + 1, iter), l2_norm_diff_LSTD_list, label='LSTD Loss', color='blue')
plt.axhline(y=Loss_BRM_real, color='red', linestyle='--', label='BRM  Loss with infinite data')
plt.axhline(y=Loss_LSTD_real, color='blue', linestyle='--', label='LSTD  Loss with infinite data')
plt.axhline(y=Loss_oracle, color='green', linestyle='--', label='Oracle Loss with Linear FA')

plt.xlabel('Number of Data Points')
plt.ylabel('L2 Norm Difference')
plt.title('Loss Curves for BRM and LSTD')
plt.legend()
plt.grid(True)
plt.show()
