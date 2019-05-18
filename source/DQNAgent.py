'''
contains the artictecture for the DQN Agent, including the neural network model
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import makeMove
import constants
import random


# board size is (8, 8)
# illegal moves will just return the initial state upon attempting to take the action

class DQNAgent:
    def __init__(self, currentState, player):
        self.memory = None # this will eventually be initialized to a numpy array
        self.currentState = currentState
        self.model = self.createModel(65, 24)
        self.epsilon = 1.0
        self.current_training_episodes = 0
        self.max_training_episodes = constants.MAX_TRAINING_EPISODES
        self.max_agent_live_episodes = constants.MAX_AGENT_LIVE_EPISODES
        self.player = player




    '''
    initializes the structure of the tensorflow feed-forward nn model needed for the dqn
    hidden_layer_size: int, input_layer_size: int -> tensorflow sequential model
    '''

    #todo: you need to somehow flatten the state structure into a 1D array of size 65
    def createModel(self, input_layer_size, hidden_layer_size):
        model = keras.Sequential()
        model.add(tf.keras.layers.Dense(input_layer_size, input_shape = (input_layer_size,), activation = 'relu'))
        model.add(tf.layers.Dense(hidden_layer_size, activation = 'relu'))
        model.add(tf.layers.Dense(hidden_layer_size, activation = 'relu'))
        model.add(tf.layers.Dense(96, activation = 'linear'))
        opt = tf.train.AdamOptimizer(learning_rate=constants.LEARNING_RATE)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        return model


    '''
    preforms memory replay on the model 
    target_network: DQN representing the active agent's target Q values
    '''
    def memory_replay(self, target_network):
        # create a minibatch of size 64
        memory_sample = np.random.choice(self.memory, 64)
        # get the target Q values for all actions
        for initial_state, action, reward, final_state in memory_sample:
            # for the next line of code, you need to ensure that you are
            next_state_return_est = constants.DISCOUNT_FACTOR * max(target_network.predict(final_state)) + reward
            return_estimation = self.model.predict(initial_state)
            return_estimation[action] = next_state_return_est
            self.model.fit(initial_state, return_estimation)


    '''
    use an epsilon greedy policy to get the next move
    returns the number of the next action
    '''
    def get_next_action(self):
        if random.uniform(0,1) <= self.epsilon: # take a random move
            nextAction = random.randint(0, 95)
            print("random move: ", nextAction)
        else:
            print(self.currentState.flatten().shape)
            print(self.currentState.flatten())
            nextAction = np.argmax(self.model.predict(self.currentState.flatten()))


        # update the explore/exploit chance
        if self.epsilon > constants.MIN_EPSILON_VALUE:
            self.epsilon *= constants.EPSILON_DECAY_RATE
        return nextAction

    '''
    adds a new memory to the agent's memory.
    if the memory is already full, the first memory is removed
    '''
    def add(self, new_memory):
        if self.memory is None:
            self.memory = np.array([new_memory])
        else:
            self.memory = np.vstack([self.memory, new_memory])
            if len(self.memory) > constants.MAX_MEMORY_CAPACITY:
                self.memory = self.memory[1:, :]



