from source import *
from source.DQNAgent import DQNAgent
from source import constants
from source.makeMove import State
from source.makeMove import get_next_state
import numpy as np

'''
contains the logic for training the agent(ie. the training loop) and allowing a human to interact with the game
'''


'''
board representation: 
index 0 is top left
index 63 is bottom right
'''

def get_initial_state(playerTurn):
    initial_board= np.array([0, -4, 0, -3, 0, -2, 0, -1])
    initial_board = np.vstack([initial_board, np.array([-8, 0, -7, 0, -6, 0, -5, 0])])
    initial_board = np.vstack([initial_board, np.array([0, -12, 0, -11, 0, -10, 0, -9])])
    for i in range(6):
        initial_board = np.vstack([initial_board, np.zeros(shape=(8,), dtype=int)])
    initial_board = np.vstack([initial_board, np.array([9, 0, 10, 0, 11, 0, 12, 0])])
    initial_board = np.vstack([initial_board, np.array([0, 5, 0, 6, 0, 7, 0, 8])])
    initial_board = np.vstack([initial_board, np.array([1, 0, 2, 0, 3, 0, 4, 0])])

    if playerTurn == constants.PLAYER1:
        return State(initial_board, constants.PLAYER1)
    else:
        return State(initial_board, constants.PLAYER2)


def train_model():
    # note: the frozen and active networks represent opposite players at ALL TIMES !!!!

    # initialize each agent
    active_network = {"training": DQNAgent(get_initial_state(constants.PLAYER1), constants.PLAYER1),
                      "target": DQNAgent(get_initial_state(constants.PLAYER1), constants.PLAYER1)}
    frozen_network = {"training": DQNAgent(get_initial_state(constants.PLAYER2), constants.PLAYER2),
                      "target": DQNAgent(get_initial_state(constants.PLAYER2), constants.PLAYER2)}


    agent_live_episodes = 0 # number of episodes a given agent has been active since last being dead/frozen
    for episode_number in range(constants.MAX_EPISODES):

        ######### episode iteration here #######
        done = False
        while not done:
            current_state = active_network["training"].currentState


            # A[k] case(epsilon greedy) -- player not waiting for opponent
            if current_state.playerTurn == active_network["training"].player:
                action = active_network["training"].get_next_action()
                done, initial_state, action, final_state, reward = get_next_state(current_state, action)
                active_network["training"].add((done, initial_state, action, final_state, reward))
                current_state = final_state #note that get_next_state must change the player's turn it is

            # waiting for opponent to make move-- must take A[97] and predict the opponent's best action
            # this is to maintain the staticness of the environment
            # note that because you are using a predefined policy here you must take that into account in your testing function
            else:
                action = np.argmax(frozen_network["target"].model.predict(current_state.flatten()))
                done, initial_state, _, final_state, reward = get_next_state(current_state, action)
                current_state = final_state
                reward *= -1
                active_network["training"].add((done, initial_state, 96, final_state, reward))

            # note that you will probably have to update parameters here regarding whose turn it is here/
            # make sure you are exact about that -- could significantly mess up the training

        active_network["training"].current_training_episodes += 1
        agent_live_episodes += 1


        # reset the current state
        # note: you probably shouldn't need to reset the target network's current state
        # that will depend on the 1 episode training logic
        active_network["training"].currentState = get_initial_state(active_network["training"].player)

        # fit the model using memory replay-- you might want to just do nothing until you have at least 64 samples
        # could just use an if statement here(would open the potential for over-fitting)
        if not len(active_network["training"].memory) < 64:
            active_network["training"].memory_replay(frozen_network["target"])

        # swap target and training network case
        if active_network["training"].current_training_episodes > \
            active_network["training"].max_training_episodes:

            # swap target and training networks and possibly update some of the agent's fields(tbd)
            active_network["training"].current_training_episodes = 0
            swap_networks(active_network["training"], active_network["target"])

        # swap active and frozen network case
        if agent_live_episodes > active_network["training"].max_agent_live_episodes:
            # swap acitve and frozen networks and possibly update the max agent live episodes field
            swap_networks(active_network["training"], frozen_network["training"])
            swap_networks(active_network["target"], frozen_network["target"])
            agent_live_episodes = 0






def play_checkers():
    pass

def main():
    pass
