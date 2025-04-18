from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
from tensorflow.keras.losses import huber
import numpy as np
import random

# Implements neural networks used in double DQN
class DNNNet:
    def __init__(self, initial_state, input_shape, num_actions, lr=0.001):
        self.lr = lr
        self.initial_state = initial_state

        # Instantiates CNN architecture
        self.model = models.Sequential()

        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(num_actions, activation='linear'))

        opt = optimizers.Adam(learning_rate=lr)
        self.model.compile(loss=huber,
                           optimizer=opt,
                           metrics=['accuracy'])
        
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def predict(self, state):
        state = self.compress_state(state)
        return self.model.predict(state, verbose=0)
    
    def fit(self, state, prediction, epochs):
        state = self.compress_state(state)
        self.model.fit(state, np.array([prediction]), verbose=0,
                       epochs=1)
        
    # Compresses picture into a 84 * 84 greyscale representation
    def compress_state(self, state):
        state = tf.expand_dims(state, axis=0)
        return state
    

class DoubleDQN:
    def __init__(self, env, c=100,
                 memory_size=100, epsilon=1,
                 sample_size=10, gamma=0.99,
                 penalty=False, d1=None,
                 d2=None):
        
        # Saves environment
        self.env = env
        model_input_shape = 8
        num_actions = env.action_space.n
        state = env.reset()

        # Implements agents memory
        self.memory = [None for _ in range(memory_size)]
        self.memory_size = memory_size
        self.memory_index = 0

        # Determines how many transitions from memory are used for learning
        self.sample_size = sample_size

        # Instantiates both neural networks
        if d1==None:
            self.q1 = DNNNet(state, model_input_shape, num_actions)
        else:
            self.q1 = d1
        if d2==None:
            self.q2 = DNNNet(state, model_input_shape, num_actions)
        else:
            self.q2 = d2

        
        # Sets neural network parameters as the same
        self.q2.set_weights(self.q1.get_weights())

        # Time steps after which second neural network weights are updated
        self.c = c

        self.epsilon = epsilon
        self.gamma = gamma

        # Sets frames after which agent is allowed to react
        # self.k = 4

        # Penalizes no game movement
        self.penalty = penalty

    def policy(self, state):
        # explores a random policy
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        # exploids current policy
        else:
            predictions = self.q1.predict(state)[0]
            return np.argmax(predictions)


    def add_to_memory(self, transition):
        self.memory[self.memory_index] = transition
        self.memory_index = (self.memory_index + 1) % self.memory_size


    def run_episode(self):

        print("-->Episode")
        state = self.env.reset()[0]

        terminated, truncated = False, False

        # Used to implement time limit for episode
        t_counter = 1
        memory_counter = 0
        sum_rewards = 0

        # Exits episode if game is terminated or time limit is reached
        while not terminated and t_counter < 400:
            t_counter += 1

            if t_counter %20 == 0:
                print(f"---->steps: {t_counter} {sum_rewards}")

            action = self.policy(state)
            next_state, reward, terminated, _, _ = self.env.step(action)
            memory_counter += 1


            sum_rewards += reward

            if self.penalty:
                reward -= 1

            # Transition is added to memory for later learning
            transition = (state, action, reward, next_state, terminated)
            self.add_to_memory(transition)

            state = next_state



            # Skip gradient descent if memory is still empty
            if t_counter <= self.sample_size:
                continue

            # Sample transitions from memory
            if t_counter > self.sample_size:
                for (mem_trans, mem_action, mem_reward,
                     mem_next_trans, mem_terminated) in random.sample(self.memory[:memory_counter],
                                                                      self.sample_size):
                    

                    
                    q_y = mem_reward + self.gamma * np.max(self.q2.predict(mem_next_trans)[0])

                    prediction = self.q1.predict(mem_trans)

                    # Adjusts Q value of prediction to increase performance
                    prediction[0][mem_action] = q_y

                    # Performs 1 step of gradient descent with adjusted prediction values
                    self.q1.fit(mem_trans, prediction, epochs=1)



            # Updates weights of second neural network
            if t_counter % (self.c-1) == 0:
                self.q2.set_weights(self.q1.get_weights())


        # Updates weights of second neural network at end of episode
        self.q2.set_weights(self.q1.get_weights())


        return sum_rewards