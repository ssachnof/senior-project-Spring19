from source import *
from source.DQNAgent import DQNAgent
from source import constants
from source.makeMove import State
import numpy as np

'''
contains the logic for training the agent(ie. the training loop) and allowing a human to interact with the game
'''

def get_initial_state():
    initial_board = []
    initial_board.append([0, -4, 0, -3, 0, -2, 0, -1])
    initial_board.append([-8, 0, -7, 0, -6, 0, -5, 0])
    initial_board.append([0, -12, 0, -11, 0, -10, 0, -9])
    for i in range(6):
        initial_board.append([0, 0, 0, 0, 0, 0, 0, 0])
    initial_board.append([9, 0, 10, 0, 11, 0, 12, 0])
    initial_board.append([0, 5, 0, 6, 0, 7, 0, 8])
    initial_board.append([1, 0, 2, 0, 3, 0, 4, 0])
    return State(initial_board, constants.PLAYER1)


def train_model():
    # initialize each agent
    active_network = {"training": DQNAgent(get_initial_state()),
                      "target": DQNAgent(get_initial_state())}
    frozen_network = {"training": DQNAgent(get_initial_state()),
                      "target": DQNAgent(get_initial_state())}

    # fill each agent's memory up to capacity
    # note: idk if you want to call fill memory on training and target networks separately
    fill_memory(active_network)
    fill_memory(frozen_network)

    agent_live_episodes = 0 # number of episodes a given agent has been active since last being dead/frozen
    for episode_number in range(constants.MAX_EPISODES):

        #########
        done = False
        while not done:
            '''
            logic for 1 episode of training 
            goes here 
            '''
        #########

        # reset the current state
        # note: you probably shouldn't need to reset the target network's current state
        # that will depend on the 1 episode training logic
        active_network["training"].currentState = get_initial_state()

        # swap target and training network case
        if active_network["training"].current_training_episodes > \
            active_network["training"].max_training_episodes:

            # swap target and training networks and possibly update some of the agent's fields(tbd)
            ...

        # swap active and frozen network case
        if agent_live_episodes > active_network["training"].max_agent_live_episodes:

            # swap acitve and frozen networks and possibly update the max agent live episodes field
            ...




'''
fills the initial memory of the agent up to capacity
not entirely sure yet if should handle the target and training networks together or separately
'''
def fill_memory(dqn):
    pass


def play_checkers():
    pass

def main():
    pass
