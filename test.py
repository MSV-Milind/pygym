from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from pybullet_env import PyBulletEnv
import time

env = PyBulletEnv(gui=True)

env = DummyVecEnv([lambda: env])

# Loading the trained model
model = PPO.load("models/npc_navigation_model.zip", env=env)

# Evaluating model in GUI mode
obs = env.reset()
for _ in range(1000): 
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break

# Rendering the final result
env.render(mode='human')
