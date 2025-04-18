# OpenAI reinforcement learning environment
import gym
import matplotlib.pyplot as plt
from celluloid import Camera
import tensorflow as tf



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