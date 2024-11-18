import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from tqdm import tqdm


'''Parameters'''

s = 3 # Size of P and Phi
feature_dim = 2 # Feature dimension
num_actions = 2 # Number of actions
gamma = 0.95 # discounting factor
# repeat = 30
# num_data_points = 25
# iter = int(n /num_data_points)

# Set random seed for reproducibility
seed = np.random.randint(0, 10000)
seed = 7272
np.random.seed(seed)
print(f"Random seed: {seed}")

'''
Random generate Phi as a (s*num_actions)xd matrix
'''
Phi = np.random.uniform(-1, 1, size=(s * num_actions, feature_dim))


# Generate a deterministic transition matrix P from (s, a) to s_prim
P = np.zeros((s * num_actions, s))
for state in range(s):
    for action in range(num_actions):
        next_state = np.random.choice(s)
        P[state * num_actions + action, next_state] = 1


theta_real = np.random.uniform(-1, 1, size=(feature_dim, 1))
Q = Phi @ theta_real 
Q += 1e-2 * np.random.randn(*Q.shape)  # Add small noise to Q, now this Q function may not be linear realizable

# Generate policy matrix pi from Q
pi = np.zeros((s, num_actions))
for state in range(s):
    action_values = Q[state * num_actions:(state + 1) * num_actions]
    best_action = np.argmax(action_values)
    pi[state, best_action] = 1
# Add noise to the policy matrix pi
noise = np.random.uniform(-0.1, 0.1, size=pi.shape)
pi_noisy = pi + noise
pi_noisy = np.clip(pi_noisy, 0, None)  # Ensure no negative probabilities
pi_noisy = pi_noisy / pi_noisy.sum(axis=1, keepdims=True)
pi = pi_noisy

'''Now we have real Q function and one noisy policy pi'''
# Generate the policy transition matrix P_pi
P_pi = np.zeros((s * num_actions, s * num_actions))
for state in range(s):
    for action in range(num_actions):
        for next_state in range(s):
            for next_action in range(num_actions):
                P_pi[state * num_actions + action, next_state * num_actions + next_action] = (
                    P[state * num_actions + action, next_state] * pi[next_state, next_action]
                )

R = Q - gamma * (P_pi @ Q)
'''Now we have the policy transition matrix P_pi and reward matrix R'''


'''Consider population distribution d, maybe off-policy'''
# Generate a random distribution over the state and action set
d = np.random.uniform(0, 1, size=(s * num_actions))
d = d / d.sum()  # Normalize to make it a valid probability distribution
# Make d a diagonal matrix
D = np.diag(d)



'''Calculate some matrix in population level'''
Sigma_cov_real = np.zeros((feature_dim, feature_dim))
for i in range(s * num_actions):
    Sigma_cov_real += d[i] * np.outer(Phi[i], Phi[i])

Sigma_cr_real = np.zeros((feature_dim, feature_dim))
for i in range(s * num_actions):
    for j in range(s * num_actions):
        Sigma_cr_real += d[i] * P_pi[i, j] * np.outer(Phi[i], Phi[j])

Sigma_next_real = np.zeros((feature_dim, feature_dim))
for i in range(s * num_actions):
    for j in range(s * num_actions):
        Sigma_next_real += d[i] * P_pi[i, j] * np.outer(Phi[j], Phi[j])

theta_phi_r_real = np.zeros((feature_dim, 1))
for i in range(s * num_actions):
    theta_phi_r_real += d[i] * (Phi[i][:, np.newaxis] * R[i])

theta_phi_prime_r_real = np.zeros((feature_dim, 1))
for i in range(s * num_actions):
    for j in range(s * num_actions):
        theta_phi_prime_r_real += d[i] * P_pi[i, j] * (Phi[j][:, np.newaxis] * R[i])
'''Calculate the best linear estimator theta_star'''
theta_star = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ Q
Q_star = Phi @ theta_star
Loss_oracle = np.linalg.norm(Q - Q_star, ord=2)
'''estimator with infinite data points'''
theta_hat_BRM_real = np.linalg.inv(Sigma_cov_real - gamma * Sigma_cr_real - gamma * Sigma_cr_real.T + gamma**2 * Sigma_next_real) @ (theta_phi_r_real - gamma * theta_phi_prime_r_real)
Q_hat_BRM_real = Phi @ theta_hat_BRM_real
Loss_BRM_real = np.linalg.norm(Q - Q_hat_BRM_real, ord=2)
theta_hat_LSTD_real = np.linalg.inv(Sigma_cov_real - gamma * Sigma_cr_real) @ (theta_phi_r_real)
Q_hat_LSTD_real = Phi @ theta_hat_LSTD_real
Loss_LSTD_real = np.linalg.norm(Q - Q_hat_LSTD_real, ord=2)
'''alpha'''
alpha_LSTD = Loss_LSTD_real/Loss_oracle
alpha_BRM = Loss_BRM_real/Loss_oracle

repeat = 5
'''Generate data'''
n = 10000
num_FQI = 50
iter = int( n / 100 )
loss_LSTD = [0] * int(n / iter)
loss_BRM = [0] * int(n / iter)
loss_BRM_SGD = [0] * int(n / iter)
loss_FQI = [0] * int(n / iter)

for _ in tqdm(range(repeat)):
    data = []
    for _ in range(n):
        state_action = np.random.choice(s*num_actions, p = d)
        phi_sa = Phi[state_action]
        noise = np.random.normal(0, 1)
        noise = 0
        reward = float(R[state_action].item() + noise)
        next_state = np.random.choice(s, p=P[state_action])
        next_action = np.random.choice(num_actions, p=pi[next_state])
        phi_sa_prime = Phi[next_state * num_actions + next_action]
        data.append((phi_sa, reward , phi_sa_prime)) 

    # Convert data to numpy array for easier manipulation
    phi_sa = np.array([d[0] for d in data])
    r_sa = np.array([d[1] for d in data])
    phi_sa_prime = np.array([d[2] for d in data])
    # print(phi_sa.shape, r_sa.shape, phi_sa_prime.shape)

    l2_norm_diff_BRM_list = []
    l2_norm_diff_LSTD_list = []
    l2_norm_diff_FQI_list = []
    for m in range(iter, n  + 1, iter):
        # Use first m data points
        phi_sa_m = phi_sa[:m]
        r_sa_m = r_sa[:m]
        phi_sa_prime_m = phi_sa_prime[:m]
        
        # Calculate covariance matrices for first m data points
        Sigma_cov_m = np.zeros((feature_dim, feature_dim))
        for phi in phi_sa_m:
            Sigma_cov_m += np.outer(phi, phi)
        Sigma_cov_m /= m
        
        Sigma_cr_m = np.zeros((feature_dim, feature_dim))
        for phi, phi_prime in zip(phi_sa_m, phi_sa_prime_m):
            Sigma_cr_m += np.outer(phi, phi_prime)
        Sigma_cr_m /= m
        
        Sigma_next_m = np.zeros((feature_dim, feature_dim))
        for phi_prime in phi_sa_prime_m:
            Sigma_next_m += np.outer(phi_prime, phi_prime)
        Sigma_next_m /= m
        
        theta_phi_r_m = np.zeros((feature_dim, 1))
        for phi, r in zip(phi_sa_m, r_sa_m):
            theta_phi_r_m += (phi[:, np.newaxis] * r)
        theta_phi_r_m /= m
        
        theta_phi_prime_r_m = np.zeros((feature_dim, 1))
        for phi_prime, r in zip(phi_sa_prime_m, r_sa_m):
            theta_phi_prime_r_m += (phi_prime[:, np.newaxis] * r)
        theta_phi_prime_r_m /= m
        
        # BRM estimator for first m data points
        theta_hat_BRM_m = np.linalg.inv(Sigma_cov_m - gamma * Sigma_cr_m - gamma * Sigma_cr_m.T + gamma**2 * Sigma_next_m) @ (theta_phi_r_m - gamma * theta_phi_prime_r_m)
        Q_hat_BRM_m = Phi @ theta_hat_BRM_m
        l2_norm_diff_BRM_m = np.linalg.norm(Q - Q_hat_BRM_m, ord=2)

        l2_norm_diff_BRM_list.append(l2_norm_diff_BRM_m)
        
        # LSTD estimator for first m data points
        theta_hat_LSTD_m = np.linalg.inv(Sigma_cov_m - gamma * Sigma_cr_m) @ (theta_phi_r_m)
        Q_hat_LSTD_m = Phi @ theta_hat_LSTD_m
        l2_norm_diff_LSTD_m = np.linalg.norm(Q - Q_hat_LSTD_m, ord=2)

        l2_norm_diff_LSTD_list.append(l2_norm_diff_LSTD_m)

        '''FQI'''
        theta_FQI = np.random.randn(feature_dim, 1)
        for _ in range(num_FQI):
            # use m data to update theta
            theta_FQI = np.linalg.inv( Sigma_cov_m) @ ( theta_phi_r_m+  gamma * Sigma_cr_m @ theta_FQI)
        Q_hat_FQI_m = Phi @ theta_FQI
        l2_norm_diff_FQI_m = np.linalg.norm(Q - Q_hat_FQI_m, ord=2)
        l2_norm_diff_FQI_list.append(l2_norm_diff_FQI_m)

    loss_LSTD = [a + b for a, b in zip(loss_LSTD, l2_norm_diff_LSTD_list)]
    loss_BRM = [a + b for a, b in zip(loss_BRM, l2_norm_diff_BRM_list)]
    loss_FQI = [a + b for a, b in zip(loss_FQI, l2_norm_diff_FQI_list)]
    '''SGD update for BRM'''
    # Initialize theta for SGD
    theta_sgd = np.random.randn(feature_dim, 1)
    learning_rate = 0.01

    l2_norm_diff_SGD_list = []

    for m in range(iter, n + 1, iter):
        for i in range(m - iter, m):
            phi_i = phi_sa[i][:, np.newaxis]
            r_i = r_sa[i]
            phi_i_prime = phi_sa_prime[i][:, np.newaxis]
            x_i = phi_i - gamma * phi_i_prime
            
            gradient = -2 * (r_i - np.dot(x_i.T, theta_sgd)) * x_i
            theta_sgd -= learning_rate * gradient
        
        Q_hat_SGD_m = Phi @ theta_sgd
        l2_norm_diff_SGD_m = np.linalg.norm(Q - Q_hat_SGD_m, ord=2)
        l2_norm_diff_SGD_list.append(l2_norm_diff_SGD_m)
    loss_BRM_SGD = [a + b for a, b in zip(loss_BRM_SGD, l2_norm_diff_SGD_list)]



loss_LSTD = [value / repeat for value in loss_LSTD]
loss_BRM = [value / repeat for value in loss_BRM]
loss_BRM_SGD = [value / repeat for value in loss_BRM_SGD]
loss_FQI = [value / repeat for value in loss_FQI]
# print(loss_FQI)


# Plot loss curves with log scale on y axis

plt.figure(figsize=(10, 6))
plt.text(n, Loss_LSTD_real, f'alpha_LSTD = {alpha_LSTD:.2f}', color='blue', verticalalignment='bottom')
plt.text(n, Loss_BRM_real, f'alpha_BRM = {alpha_BRM:.2f}', color='red', verticalalignment='bottom')
plt.plot(range(iter, n + 1, iter), loss_FQI, label='FQI Loss', color='black')
plt.plot(range(iter, n + 1, iter), loss_BRM, label='BRM Loss', color='red')
plt.plot(range(iter, n + 1, iter), loss_BRM_SGD, label='BRM_SGD Loss', color='magenta')
plt.plot(range(iter, n + 1, iter), loss_LSTD,linestyle='--', label='LSTD Loss', color='blue')
plt.axhline(y=Loss_BRM_real, color='red', linestyle='--', label='BRM  Loss with infinite data')
plt.axhline(y=Loss_LSTD_real, color='blue', linestyle='--', label='LSTD  Loss with infinite data')
plt.axhline(y=Loss_oracle, color='green', linestyle='--', label='Oracle Loss with Linear FA')

plt.xlabel('Number of Data Points')
plt.ylabel('L2 Norm Difference')
plt.yscale('log')
plt.title('Loss Curves for BRM and LSTD')
plt.legend()
plt.grid(True)
plt.show()