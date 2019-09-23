'''
contains the artictecture for the DQN Agent, including the neural network model
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import makeMove
import constants
import random
import collections


# board size is (8, 8)
# illegal moves will just return the initial state upon attempting to take the action

class DQNAgent:
    def __init__(self, currentState, player):
        self.memory = None # this will eventually be initialized to a numpy array
        self.currentState = currentState
        self.model = self.createModel(65, 48)
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
        model.add(tf.keras.layers.Dropout(.2))
        #model.add(tf.layers.Dense(hidden_layer_size, activation = 'relu'))
        model.add(tf.layers.Dense(96, activation = 'linear'))
        opt = keras.optimizers.Adam()
        #opt = tf.train.AdamOptimizer(learning_rate=constants.LEARNING_RATE)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        return model


    '''
    preforms memory replay on the model 
    target_network: DQN representing the active agent's target Q values
    '''
    def memory_replay(self, target_network, max_memory_capacity, epsilon_decay_rate):
        # create a minibatch of size 64
        # print(self.memory)
        sample_size = int(.1 * len(self.memory))
        indices = np.random.choice(self.memory.shape[0], sample_size, replace=False)
        memory_sample = self.memory[indices]
        targets = []
        features = []
        # get the target Q values for all actions
        self.update_epsilon(epsilon_decay_rate, max_memory_capacity)
        for done,initial_state, action, final_state, reward in memory_sample:
            # for the next line of code, you need to ensure that you are
            next_state_return_est = constants.DISCOUNT_FACTOR * max(target_network.model.predict(final_state.flatten())[0]) + reward
            if done:
                next_state_return_est = reward
            return_estimation = self.model.predict(initial_state.flatten())
            return_estimation[0][action] = next_state_return_est
            targets.append(return_estimation[0])
            features.append(initial_state.flatten()[0])#may want to change this to just append the board
        targets = np.array(targets)
        features = np.array(features)
        self.model.fit(features, targets, verbose=1, validation_split=1)
        # if self.epsilon > constants.MIN_EPSILON_VALUE:
        #     self.epsilon *= constants.EPSILON_DECAY_RATE
        print("EPSILON: ", self.epsilon)
        print("Memory replay completed!!!!!!!")
            # print("fitted!!!!!!!!")


    '''
    use an epsilon greedy policy to get the next move
    returns the number of the next action
    '''
    def get_next_action(self, max_memory_size, legal_moves, distanceFromBest = 0):
        print('lm: ', legal_moves)
        lm = list(legal_moves)
        if random.uniform(0,1) <= self.epsilon: # take a random move
            # nextAction = random.randint(0, 95)
            nextAction = random.choice(lm)
            return nextAction, distanceFromBest
            # distanceFromBest = 0

            # print("random move: ", nextAction)
        else:
            # print(self.currentState.flatten().shape)
            # print(self.currentState.flatten())

            # return the best legal move to make
            allActions = self.model.predict(self.currentState.flatten())[0]
            q_values = []
            for lm_i in lm:
                q_values.append(allActions[lm[lm_i]])
            return lm[np.argmax(np.array(q_values))], distanceFromBest



        # # update the explore/exploit chance
        # # todo: this needs to only be updated after a episode has finished
        # # if self.memory is not None:
        # #     print(len(self.memory))
        # # if self.epsilon > constants.MIN_EPSILON_VALUE and self.memory is not None and \
        # #         len(self.memory) == max_memory_size:
        # #     self.epsilon *= epsilon_decay_rate
        # results = self.model.predict(self.currentState.flatten())[0]
        # results = np.sort(results.flatten())[-1::-1]
        # valueToFind = results[distanceFromBest]
        #
        # predicted_values = self.model.predict(self.currentState.flatten())[0]
        # # print(predicted_values)
        # # print(valueToFind)
        # # print(np.max(predicted_values))
        # nextAction = np.where(predicted_values == valueToFind)[0]
        # if distanceFromBest + 1 >= 96:
        #     exit("!!!!!!!!!!!!!!DISTANCE FROM BEST TOO LARGE!!!!!!!!!!!!!")
        # return nextAction, distanceFromBest + 1




    def update_epsilon(self, epsilon_decay_rate, max_memory_size):
        if self.epsilon > constants.MIN_EPSILON_VALUE and self.memory is not None and \
                len(self.memory) == max_memory_size:
            self.epsilon *= epsilon_decay_rate

    '''
    adds a new memory to the agent's memory.
    if the memory is already full, the first memory is removed
    '''
    def add(self, new_memory, max_memory_capacity, epsilon_decay_rate):
        if self.memory is None:
            self.memory = np.array([new_memory])
        else:
            # self.memory = np.append(self.memory, [new_memory], axis=0)
            if len(self.memory) >= max_memory_capacity:
                self.memory[:-1] = self.memory[1:]; self.memory[-1] = new_memory
            else:
                self.memory = np.append(self.memory, [new_memory], axis=0)


