import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np

class PyBulletEnv(gym.Env):
    def __init__(self, gui=False):
        super(PyBulletEnv, self).__init__()
        print("Initializing PyBullet environment...")
        self.gui = gui
        if self.gui:
            print("Connecting to PyBullet GUI...")
            self.physics_client = p.connect(p.GUI)
        else:
            print("Connecting to PyBullet in headless mode...")
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load environment (plane, NPC, obstacles)
        try:
            print("Loading plane URDF...")
            p.loadURDF("plane.urdf")
            print("Loading sphere URDF (NPC)...")
            self.npc_id = p.loadURDF("sphere2.urdf", [0, 0, 0.2], globalScaling=0.5)
            print("Loading obstacle URDFs...")
            self.obstacle_ids = []
            for i in range(3):  # Example: Adding 3 obstacles
                self.obstacle_ids.append(p.loadURDF("cube.urdf", [2 + i * 2, 1, 0.5], globalScaling=0.5))
        except Exception as e:
            print(f"Error loading URDFs: {e}")

        self.goal_position = np.array([5, 5, 0])

        # Define the observation space as a dictionary
        self.observation_space = spaces.Dict({
            "npc_position": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
            "obstacle_positions": spaces.Box(low=-10.0, high=10.0, shape=(len(self.obstacle_ids) * 3,), dtype=np.float32),
            "goal_position": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
        })

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.max_steps = 200
        print("PyBullet environment initialized.")

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.npc_id = p.loadURDF("sphere2.urdf", [0, 0, 0.2], globalScaling=0.5)
        self.obstacle_ids = [p.loadURDF("cube.urdf", [2 + i * 2, 1, 0.5], globalScaling=0.5) for i in range(3)]

        self.steps = 0
        npc_pos, _ = p.getBasePositionAndOrientation(self.npc_id)
        obstacle_pos = [p.getBasePositionAndOrientation(obstacle)[0] for obstacle in self.obstacle_ids]
        self.state = {
            "npc_position": np.array(npc_pos, dtype=np.float32),
            "obstacle_positions": np.array([pos for obs_pos in obstacle_pos for pos in obs_pos], dtype=np.float32),
            "goal_position": self.goal_position.astype(np.float32),
        }
        return self.state

    def step(self, action):
        self.steps += 1

        # Clip action to allowable range and calculate new position
        velocity = np.clip(action, -1, 1)
        npc_pos, _ = p.getBasePositionAndOrientation(self.npc_id)
        new_pos = np.array(npc_pos[:2]) + velocity * 0.1
        p.resetBasePositionAndOrientation(self.npc_id, np.append(new_pos, 0.2), [0, 0, 0, 1])

        # Perform one step in the simulation
        p.stepSimulation()

        # Get updated positions of NPC and obstacles
        npc_pos, _ = p.getBasePositionAndOrientation(self.npc_id)
        obstacle_pos = [p.getBasePositionAndOrientation(obstacle)[0] for obstacle in self.obstacle_ids]

        # Update state as dictionary
        self.state = {
            "npc_position": np.array(npc_pos, dtype=np.float32),
            "obstacle_positions": np.array([pos for obs_pos in obstacle_pos for pos in obs_pos], dtype=np.float32),
            "goal_position": self.goal_position.astype(np.float32),
        }

        # Calculate distance to the goal
        distance_to_goal = np.linalg.norm(self.goal_position - np.array(npc_pos))
        reward = -distance_to_goal

        # Determine if the episode is done
        done = False
        if distance_to_goal < 0.1:  # NPC reaches the goal
            reward += 10
            done = True
        if self.steps >= self.max_steps:  # Maximum steps reached
            done = True

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        print("Closing PyBullet environment...")
        p.disconnect(self.physics_client)
