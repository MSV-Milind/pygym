from stable_baselines3 import PPO
from pybullet_env import PyBulletEnv

model = PPO.load("models/npc_navigation_model")

env = PyBulletEnv(gui=False)

obs = env.reset()

# Run the evaluation loop
done = False
steps = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Step {steps}: Action taken - {action}, Reward - {reward}, Done - {done}")
    steps += 1

env.close()
