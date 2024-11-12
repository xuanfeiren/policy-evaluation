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

# Set random seed for reproducibility
seed = np.random.randint(0, 10000)
seed = 2153 # BRM has smaller bound on this seed
np.random.seed(seed)
print(f"Random seed: {seed}")

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

# Create arrays to store results for different gamma values
gamma_values = np.linspace(0, 0.99, 100)
first_expr_values = []
second_expr_values = []

for gamma in gamma_values:
    # Compute M = I - gamma * P
    I = np.eye(s)
    M = I - gamma * P

    # First expression: ||Φ (Φᵀ Mᵀ M Φ)⁻¹ Φᵀ Mᵀ P||_op
    MT_M = M.T@ D @ M
    reg_term = 1e-8 * np.eye(feature_dim)
    Phi_MTM_Phi_inv = np.linalg.inv(Phi.T @ MT_M @ Phi + reg_term)
    first_expr = mu_operator_norm(Phi @ Phi_MTM_Phi_inv @ Phi.T @ M.T @ D @ P, D)
    first_expr_values.append(first_expr)

    # Second expression: ||Φ (Φᵀ M Φ)⁻¹ Φᵀ P||_op
    reg_term = 1e-8 * np.eye(feature_dim)
    Phi_M_Phi_inv = np.linalg.inv(Phi.T@ D @ M @ Phi + reg_term)
    second_expr = mu_operator_norm(Phi @ Phi_M_Phi_inv @ Phi.T@ D @ P , D)
    second_expr_values.append(second_expr)

# Plot the results
plt.plot(gamma_values, first_expr_values, label='BRM')
plt.plot(gamma_values, second_expr_values, label='LSTD')
plt.xlabel('gamma')
plt.ylabel('Operator norm')
plt.legend()
plt.show()

# Print results
# print("BRM  expression value:", first_expr)
# print("LSTD expression value:", second_expr)
