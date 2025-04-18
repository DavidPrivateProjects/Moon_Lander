# Lunar Lander Double DQN Agent
A reinforcement learning project using a Double Deep Q-Network (Double DQN) agent to solve OpenAI Gym's LunarLander-v2 environment. The project includes real-time video rendering of agent gameplay, random agent baseline evaluation, and full training and testing of the Double DQN agent with optional performance visualization.

https://github.com/user-attachments/assets/7d5925c2-2904-4985-b3a2-a123bf6a2a00

## Features
- Implementation of a Double DQN agent from scratch using TensorFlow/Keras
- Optional penalty-based exploration to discourage inaction
- Performance video generation using celluloid
- Random agent baseline testing for comparison
- Epsilon decay for exploration-exploitation balance
- Network synchronization for stable learning
- Realtime plots and statistics of training performance

## Requirements
- Python 3.7+
- OpenAI Gym
- NumPy
- TensorFlow 2.x
- Matplotlib
- Celluloid


## Usage
1. Generate a gameplay video of a random agent
- Evaluate 100 random agents to determine baseline reward stats
- Save a plot of reward scores across agents

2. Train the Double DQN agent
- In the same script, the agent is trained for a fixed number of episodes.
- The number of episodes and training parameters can be adjusted as needed.
- Logs per-episode rewards.
- Applies epsilon-greedy exploration with linear decay.
- Trains the Q-networks using experience replay.

3. Render the trained agent
- At the end of training, a video of the trained agent will be created and saved.
  
## Customization
Following parameters can be modified in future experiments:

- DoubleDQN(...)
- epsilon: Initial exploration rate
- gamma: Discount factor
- memory_size: Experience replay buffer size
- sample_size: Mini-batch size for training
- penalty: If True, penalizes idle behavior

## Neural network structure:

- self.model.add(layers.Dense(16, activation='relu'))
- self.model.add(layers.Dense(16, activation='relu'))
- self.model.add(layers.Dense(num_actions, activation='linear'))

## How It Works
Double DQN Architecture
- Two neural networks (q1, q2) predict Q-values
- q1 is updated frequently; q2 is synchronized every c steps
- Training uses experience replay and target Q-values from q2

Epsilon-Greedy Policy
- Starts with high exploration (random actions)
- Gradually favors exploitation based on learned Q-values

Video Rendering
- Screenshots are captured using matplotlib's camera.snap()
- Video is exported using camera.animate().save(...)

License
This project is licensed under the MIT License. Feel free to use and modify the code for educational or research purposes.

