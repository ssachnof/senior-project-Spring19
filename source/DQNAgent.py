'''
contains the artictecture for the DQN Agent, including the neural network model
'''


import tensorflow as tf





class DQNAgent:
    def __init__(self, memory, currentState):
        self.memory = memory
        self.currentState = currentState
        self.model = self.createModel(64, 128, 32)



    '''
    initializes the structure of the tensorflow feed-forward nn model needed for the dqn
    hidden_layer_size: int, input_layer_size: int -> tensorflow sequential model
    '''
    def createModel(self, input_layer_size, hidden_layer_size, batch_size):
        pass

    '''
    preforms memory replay on the model 
    '''
    def memory_replay(self):
        pass


    '''
    update the neural network's weights based on the target network's Q values and the active network's Q values
    note: idk if this method is actually necessary-- tbd
    '''
    def update_weights(self, target, actual):
        pass