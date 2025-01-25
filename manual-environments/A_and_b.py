import numpy as np

# Parameters
s = 10 # Size of P and Phi
feature_dim = s # Feature dimension
gamma = 0.95 # discounting factor

# Set random seed for reproducibility
seed = np.random.randint(0, 10000)
np.random.seed(seed)
print(f"Random seed: {seed}")

# Data distribution matrix
# Create diagonal data distribution matrix D with probabilities summing to 1
d = np.random.random(s)
d = d / np.sum(d)  # Normalize to sum to 1
D = np.diag(d)

# Randomly generate P as a s*s stochastic matrix with positive entries (each row sums to 1)
# P = np.random.random((s, s))
# P = P / P.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1

# Generate a deterministic transition matrix P
P = np.zeros((s, s))
for i in range(s):
    next_state = np.random.choice(s)
    P[i, next_state] = 1
# Random generate Phi as a sxd matrix
# Phi = np.random.uniform(-10, 10, size=(s, feature_dim))

#tabular case
Phi = np.eye(s)
# Compute M = I - gamma * P
I = np.eye(s)
M = I - gamma * P
# Compute v = (I - gamma * P)^(-1) * R
theta_real = np.random.uniform(-10, 10, size=(feature_dim, 1))
v = Phi @ theta_real 
# v += 1e-1 * np.random.randn(*v.shape)  # Add small noise to v
R = v - gamma * (P @ v)



# Find theta_star to minimize ||Phi * theta - v||_2
''' Calculate real covariance matrices using the real distribution d and transition matrix P'''
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
        
''' Generate data'''

A_LSTD = Sigma_cov_real - gamma * Sigma_cr_real
b_LSTD = theta_phi_r_real

A_BRM = Sigma_cov_real - gamma * Sigma_cr_real - gamma * Sigma_cr_real.T + gamma**2 *  Sigma_next_real
b_BRM = theta_phi_r_real - gamma * theta_phi_prime_r_real

sigma_min_A_LSTD = np.min(np.linalg.eigvals(A_LSTD))
sigma_min_A_BRM = np.min(np.linalg.eigvals(A_BRM))

ratio_A = sigma_min_A_BRM / sigma_min_A_LSTD
print(f"Ratio of minimum singular values of A_LSTD and A_BRM: {ratio_A}")
ratio_b =  np.linalg.norm(b_BRM) / np.linalg.norm(b_LSTD)
print(f"Ratio of L2 norms of b_LSTD and b_BRM: {ratio_b}")
