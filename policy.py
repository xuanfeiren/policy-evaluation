import gymnasium as gym
# Add this import to ensure proper initialization
# import torch
from stable_baselines3 import PPO

# Create the Cartpole environment
# env = gym.make('CartPole-v1')
env = gym.make('MountainCar-v0')


# Instantiate the agent
# model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
# model.learn(total_timesteps=100000)

# Save the trained model
# model.save("ppo_mountaincar")

# Load the trained model
model = PPO.load("ppo_mountaincar")

# Evaluate the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()
