# OpenAI reinforcement learning environment
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
from tensorflow.keras.losses import huber
from celluloid import Camera


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Used to record videos of agent performance
def create_env_video(name, agent=None, video_len=500):

    # Instantiates environment and initial state
    env = gym.make("LunarLander-v2", render_mode='rbg_array')
    state = env.reset()

    fig = plt.figure()
    camera = Camera(fig)

    for i in range(video_len):

        # Random action is chosen
        if agent is None:
            state, reward, terminated, truncated, _ = env.step(env.action_space.sample())

        # Action is chosen by reinforcement agent
        else:
            if state[1] == {}:
                state = state[0]
            action = agent.policy(tf.expand_dims(state, axis=0))
            state, reward, terminated, truncated, _ = env.step(action)

        # Environment is restarted if game was finished during video
        if terminated or truncated:
            state = env.reset()

        # Makes a screenshot of current state
        plt.imshow(env.render())
        camera.snap()

    # Combines and saves all screenshots
    animation = camera.animate()
    animation.save(f"{name}.git", fps=40)
    plt.show()

    env.close()


if __name__ == "__main__":
    # Creates a video of the problem state using the random agent
    create_env_video("moon lander random", agent=None, video_len=500)

    # Determines average total reward a random strategy would score in env!

    # 100 agents are tested for 500 frames
    num_agents = 100
    len_test = 200

    # Instantiates environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env.reset()

    total_rewards = []

    for agent in range(num_agents):
        total_agent_rewards = 0
        env.reset()
        
        terminated = False
        
        while not terminated:
            state, reward, terminated, truncated, _ = env.step(env.action_space.sample())



            total_agent_rewards += reward

        total_rewards.append(total_agent_rewards)

    env.close()
    print("Total reward average (random agent): ", np.mean(total_rewards))
    print("Total reward std (random agent): ", np.std(total_rewards))

    plt.plot(total_rewards)
    plt.title("Random Gameplay Average Results")
    plt.xlabel("Agent")
    plt.ylabel("Total Reward")
    plt.savefig("Average reward random gameplay")


    # Trains the double DQN agent on the Lunar Lander environment
    num_agents = 1
    num_episodes = 500

    # Instantiates environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env.reset()

    now = time.time()
    agent_rewards = []

    for agent in range(0, num_agents) :

        print(f"Agent {agent + 1}:", end=' ')

        episode_rewards = []
        dqn_agent = DoubleDQN(env, penalty=False)

        # Runs all episodes
        for episode in range(0, num_episodes) :
            sum_rewards = dqn_agent.run_episode()
            episode_rewards.append(sum_rewards)
            print(episode, end=" ")
            print(sum_rewards)

            # Implements linear epsilon decay
            if dqn_agent.epsilon > 0: #Decay when epsilon > 0 only
                dqn_agent.epsilon -= 0.0018

        # Saves rewards
        agent_rewards.append(episode_rewards)
        print(f"time passed: {time.time() - now}")


    print("Seconds passed: ", time.time() - now)
    env.close()


    plt.title("Double DQN Agent Learning Results")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.plot(agent_rewards[0])
    plt.savefig("double dqn agent learning results (100 epochs)")

    print("Total reward average (DQN agent): ", np.mean(agent_rewards[0]))
    print("Total reward std (DQN agent): ", np.std(agent_rewards[0]))


    create_env_video("trained moon lander", agent=dqn_agent, video_len=500)















# create_env_video("moon lander random", agent=None, video_len=500)