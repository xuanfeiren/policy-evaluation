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
seed = 2153 # BRM has larger bound on this seed
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

# Random generate Phi as a sxd matrix
Phi = np.random.uniform(-10, 10, size=(s, feature_dim))
R = np.random.uniform(-10, 10, size=(s, 1))
# Compute M = I - gamma * P
I = np.eye(s)
M = I - gamma * P
# Compute v = (I - gamma * P)^(-1) * R

v = np.linalg.inv(I - gamma * P) @ R
# print("Real value function v:", v)
# Number of data points


n = 1000
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

iter = int(n/10)
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

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(iter, n + 1, iter), l2_norm_diff_BRM_list, label='BRM Loss')
plt.plot(range(iter, n + 1, iter), l2_norm_diff_LSTD_list, label='LSTD Loss')
plt.xlabel('Number of Data Points')
plt.ylabel('L2 Norm Difference')
plt.title('Loss Curves for BRM and LSTD')
plt.legend()
plt.grid(True)
plt.show()
