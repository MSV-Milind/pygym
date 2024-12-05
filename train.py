from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from pybullet_env import PyBulletEnv  


env = PyBulletEnv(gui=False)  

# Wrap environment
env = DummyVecEnv([lambda: env])

# Modify PPO parameters for better learning
model = PPO("MultiInputPolicy", env, verbose=1, 
            learning_rate=0.0001,   
            gamma=0.98,             
            n_steps=4096,           
            batch_size=128,         
            ent_coef=0.01,         
            vf_coef=0.5,           
            max_grad_norm=0.5)     

# Training the model
model.learn(total_timesteps=200000)  

model.save("models/npc_navigation_model.zip")
