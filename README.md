# NPC Navigation with PPO in PyBullet

Welcome to the project for training an NPC (Non-Player Character) to navigate through a simulated obstacle course using Proximal Policy Optimization (PPO) reinforcement learning in PyBullet.

This project will guide you through the setup, training, and testing of the model, providing the necessary steps to get everything running smoothly.

## Requirements

Before you begin, make sure you have the following installed:

- Python 3.8 or later
- PyBullet (`pip install pybullet`)
- Stable-Baselines3 (`pip install stable-baselines3`)
- Gym (`pip install gym`)

## Step 1: Set Up the Environment

1. **Install dependencies**:  
   Make sure you have the required libraries installed:
   ```bash
   pip install gym pybullet stable-baselines3

## Step 2: Understand the Custom Environment (`pybullet_env.py`)

The custom environment is crucial for simulating the obstacle course in PyBullet, where the NPC will be trained using PPO. Here’s how to set it up and make adjustments:

### 1. Overview of the Custom Environment

The `pybullet_env.py` file defines the environment that the NPC interacts with. This environment is based on the PyBullet simulation platform and models physical properties like collisions, gravity, and movement. The NPC's task is to navigate through an obstacle course, avoiding obstacles and moving towards a goal.

### 2. Key Components of the Environment

- **Obstacles**: The environment includes static or dynamic obstacles that the NPC needs to avoid. You can modify their sizes, shapes, and positions by editing the environment file.
- **Action and Observation Spaces**: The action space defines the possible moves the NPC can take (e.g., moving forward, turning). The observation space defines the sensory inputs, like distances to obstacles and positions in the world.
- **Reward System**: The reward is provided based on the NPC's actions, encouraging the agent to complete the course while avoiding collisions.

### 3. Modify the Environment

If you want to make changes to the environment (e.g., increase difficulty or change NPC behavior), you can adjust the following:

- **Obstacle Configuration**: Modify how obstacles are placed or how they move to make the course more challenging.
- **NPC Starting Position**: Change where the NPC starts in the environment to test different scenarios.
- **Reward Mechanism**: Adjust the rewards based on new behaviors or performance criteria.

## Step 3: Modify Training Parameters in `train.py`

In this step, you will adjust the training parameters in `train.py` to fine-tune the performance of your model. By modifying these parameters, you can control how your NPC is trained in the environment defined in Step 2.

### 1. Key Parameters to Adjust

Here are some important parameters in the `train.py` file that you can modify to optimize your training:

- **`learning_rate`**: This controls the step size the model takes during training. A lower value (e.g., 0.0001 or 0.0002) can make the training more stable but slower. A higher value can speed up learning but might make the training less stable.
  
  ```python
  learning_rate=0.0001

## Step 4: Monitor Training and Adjust Hyperparameters

During training, it's important to monitor the model's performance and adjust hyperparameters if necessary. This ensures that the NPC learns effectively and doesn't overfit or underfit to the environment.

### 1. Monitor Key Training Metrics

You can monitor the following key metrics during training:

- **`fps` (Frames Per Second)**: This shows how fast the model is interacting with the environment. A higher FPS indicates faster processing.
- **`total_timesteps`**: The cumulative number of steps the model has taken in the environment. It shows how much experience the model has accumulated.
- **`explained_variance`**: This metric indicates how well the model's value function approximates the expected future reward. Higher values suggest the model has a good understanding of the environment.
- **`loss`**: The overall loss, which combines the policy loss, value loss, and entropy loss. Monitoring this helps identify if the model is making effective progress.
- **`value_loss`**: The loss related to the value function, which helps the model evaluate the expected reward of a given state.
- **`entropy_loss`**: This controls the exploration-exploitation trade-off. A higher entropy encourages more exploration.

### 2. Evaluate Performance at Intervals

After running the model for some time, it’s a good practice to evaluate its performance:

- **Track `ep_rew_mean`**: This is the average reward per episode, which helps assess whether the model is improving its behavior over time.
- **Test the model periodically**: Once a significant number of timesteps have been processed, run the trained model in the environment to visually or programmatically assess its navigation behavior. This will help identify if the NPC is learning to navigate the obstacle course effectively.

### 3. Adjust Hyperparameters Based on Performance

Based on the results from monitoring the training metrics, you may need to adjust the hyperparameters:

- **Learning Rate**: If the model is not converging or is fluctuating too much, reduce the learning rate. A smaller learning rate makes the training more stable but slower.
- **Batch Size**: If training is too slow or the model isn't generalizing well, adjust the batch size. A larger batch size can improve stability but requires more memory and computation.
- **Gamma (Discount Factor)**: If the model is focusing too much on immediate rewards, decrease `gamma`. If it isn't considering long-term rewards, increase `gamma`.

### 4. Adjust Training Duration

- **Increase `total_timesteps`** if the model hasn’t reached satisfactory performance after the set number of timesteps. Training for more steps may give the model more time to learn the optimal policy.
- If the model's performance is plateauing, consider reducing `total_timesteps` or stopping early if it's already reaching a satisfactory level.

### 5. Fine-tuning for Better Results

After a few iterations, you may notice some trends in the training logs. If necessary, fine-tune the following:

- **Adjust the value function**: If the value loss is significantly higher than the policy loss, it may indicate that the model's value function is not accurate enough. You may need to adjust its complexity or learning rate.
- **Entropy loss**: If exploration becomes too sparse, reduce the value of `entropy_loss` to increase the model's exploration behavior.
- **Policy Gradient Loss**: If `policy_gradient_loss` is too high, you may want to adjust the optimization strategy or learning rate.

By monitoring these metrics and making adjustments as needed, you can ensure your model learns effectively and performs well in the task of navigating the obstacle course.

### 6. Save the Model

Once you've achieved satisfactory performance, save the trained model using:

```python
model.save("models/npc_navigation_model.zip")



