from source import *
from source.DQNAgent import DQNAgent
from source import constants
from source.makeMove import State
from source.makeMove import get_next_state
from source.makeMove import get_all_legal_moves
import numpy as np
import copy
import tensorflow
from tensorflow.keras import models
import matplotlib.pyplot as plt
import math
import pickle
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
    for i in range(2):
        initial_board = np.vstack([initial_board, np.zeros(shape=(8,), dtype=int)])
    initial_board = np.vstack([initial_board, np.array([9, 0, 10, 0, 11, 0, 12, 0])])
    initial_board = np.vstack([initial_board, np.array([0, 5, 0, 6, 0, 7, 0, 8])])
    initial_board = np.vstack([initial_board, np.array([1, 0, 2, 0, 3, 0, 4, 0])])
    if playerTurn == constants.PLAYER1:
        return State(initial_board, constants.PLAYER1)
    else:
        return State(initial_board, constants.PLAYER2)


def train_model(max_live_episodes, max_training_episodes, max_memory_capacity, epsilon_decay_rate):
    # note: the frozen and active networks represent opposite players at ALL TIMES !!!!

    # initialize each agent
    active_network = {"training": DQNAgent(get_initial_state(constants.PLAYER1), constants.PLAYER1),
                      "target": DQNAgent(get_initial_state(constants.PLAYER1), constants.PLAYER1)}
    frozen_network = {"training": DQNAgent(get_initial_state(constants.PLAYER2), constants.PLAYER2),
                      "target": DQNAgent(get_initial_state(constants.PLAYER2), constants.PLAYER2)}


    agent_live_episodes = 0 # number of episodes a given agent has been active since last being dead/frozen

    plot_x = []
    plot_y = []
    min_epsilon = .1
    plt.xlabel('Episode')
    plt.ylabel('Consecutive Moves')
    plt.title("live episodes: " + str(max_live_episodes) + " training episodes: " +
              str(max_training_episodes) + "\n mem cap: " + str(max_memory_capacity) + ' epsilon decay rate: ' + str(epsilon_decay_rate))
    for episode_number in range(constants.MAX_EPISODES):
        if episode_number > 7500:
            # if active_network['training'].epsilon <= min_epsilon and active_network['target'].epsilon <= min_epsilon and\
            #     frozen_network['training'].epsilon <= min_epsilon and frozen_network['target'].epsilon <= min_epsilon and\
            #         episode_number > 10000:
            plt.plot(plot_x, plot_y)
            plt.savefig(str(max_live_episodes) + "-" + str(max_training_episodes) + "-" + str(max_memory_capacity) + "-" +
                        str(epsilon_decay_rate) + ".png")
            plt.close()
            print("saving figure to path " + str(max_live_episodes) + "-" + str(max_training_episodes) + "-" +
                  str(max_memory_capacity) + '-' + str(epsilon_decay_rate) + ".png")
            if frozen_network['target'].player == constants.PLAYER2:
                frozen_network['target'].model.save('p2_weights.h5')
                active_network['target'].model.save('p1_weights.h5')
            else:
                frozen_network['target'].model.save('p1_weights.h5')
                active_network['target'].model.save('p2_weights.h5')
            break

        print("EPISODE_NUMBER: ", episode_number)

        ######### episode iteration here #######
        done = False
        current_state = active_network["training"].currentState
        consecutive_moves = 0
        while not done:
            consecutive_moves += 1
            current_state = active_network["training"].currentState
            # print()
            # print("turn: ", current_state.playerTurn)
            # legal_moves = get_all_legal_moves(current_state)
            # print('pt: ', current_state.playerTurn)
            # print('lm: ', legal_moves)
            # print(current_state.board)
            action, _ = active_network["training"].get_next_action(max_memory_capacity, None, 0)
            # print(current_state.playerTurn)
            done, initial_state, action, intermediate_state, reward = get_next_state(current_state, active_network['training'], max_memory_capacity, 0, action, None)
            # print(intermediate_state.playerTurn)
            # exit()
            # print('in_s: \n\n',initial_state.board)
            # print('im_s: \n\n', intermediate_state.board)
            #need to immitate that opp saw something and couldn't make a move-- just have to swap the player's turn it is
            if done:
                intermediate_state.playerTurn *= -1
                active_network["training"].add((done, initial_state, action, intermediate_state, reward), max_memory_capacity,
                                               epsilon_decay_rate)
                if not len(active_network["training"].memory) < max_memory_capacity:
                    active_network["training"].memory_replay(frozen_network["target"], max_memory_capacity, epsilon_decay_rate)
                break
                # if reward != -2:
                #     print('REWARD: ', reward)
                #     # important note: it looks like your final board state is intermediate state, not final state
                #     print('final board: \n', active_network['training'].currentState.board, "\n\n ", intermediate_state.board)
                #     exit("SUCCESS!!!!!!")
                # break
            else:
                legal_moves = get_all_legal_moves(intermediate_state)
                consecutive_moves+=1
                # print(current_state.board)
                # opp_action = np.argmax(frozen_network["target"].model.predict(intermediate_state.flatten()))#note that you will need to change s.t. a valid move is chosen
                opp_action, _ = frozen_network["target"].get_next_action(max_memory_capacity, legal_moves, 0)
                done, _, opp_action, final_state, reward = get_next_state(intermediate_state, frozen_network['target'], max_memory_capacity, 0, opp_action, legal_moves)
                # print('fs_s: \n\n', final_state.board)
                # print(final_state.board)
                # this not needed because eventually, the opponent will learn to only make valid moves
                # however, not including it may slow down training, but unsure if including it
                # will mess up the training process

                # may need to stop cheating, seems to be causing a plateau
                # if reward == -2:
                #     reward = 0
                reward *= -1
                active_network["training"].currentState = final_state
                active_network["training"].add((done, initial_state, action, final_state, reward), max_memory_capacity,
                                               epsilon_decay_rate)
                if not len(active_network["training"].memory) < max_memory_capacity:
                    active_network["training"].memory_replay(frozen_network["target"], max_memory_capacity, epsilon_decay_rate)

                # print('DONE: ', done)
                # print(final_state.board)
                # print('DONE')

            # if done and reward == 0:
            #     print("tie game")
            #     print(current_state.board)
            #     exit()
            # if not done:
            #     print("turn: ", current_state.playerTurn)
            #     print("struct current turn: ", active_network["training"].currentState.playerTurn)

            # note that you will probably have to update parameters here regarding whose turn it is here/
            # make sure you are exact about that -- could significantly mess up the training
        done = False
        # print("END OF EPISODE!!!!!")
        active_network["training"].current_training_episodes += 1
        agent_live_episodes += 1
        print()
        print("CONSECUTIVE MOVES: ", consecutive_moves)


        # reset the current state
        # note: you probably shouldn't need to reset the target network's current state
        # that will depend on the 1 episode training logic
        active_network["training"].currentState = get_initial_state(active_network["training"].player)

        # fit the model using memory replay-- you might want to just do nothing until you have at least 64 samples
        # could just use an if statement here(would open the potential for over-fitting)
        if not len(active_network["training"].memory) < max_memory_capacity:
            plot_x.append(episode_number)
            plot_y.append(consecutive_moves)
            # note that memory replay already occured
            # active_network["training"].memory_replay(frozen_network["target"], max_memory_capacity, epsilon_decay_rate)

        # swap target and training network case
        # if active_network["training"].current_training_episodes > \
        #     active_network["training"].max_training_episodes:
        correct_swapping = False
        # if active_network["training"].current_training_episodes >= max_training_episodes:
        if (episode_number + 1) % max_training_episodes == 0:
            print("\n\n\n#########SWAPPING TARGET AND TRAINING############\n\n\n")
            # exit("swapping target and training")

            # swap target and training networks and possibly update some of the agent's fields(tbd)
            active_network["training"].current_training_episodes = 0
            # print("target epsilon: ", active_network["target"].epsilon)
            active, frozen = swap_networks(active_network["training"], active_network["target"])
            active_network["training"] = active
            active_network["target"] = frozen
            # print("training epsilon: ", active_network["training"].epsilon)
            # print("target epsilon: ", active_network["target"].epsilon)

        # swap active and frozen network case
        # if agent_live_episodes > active_network["training"].max_agent_live_episodes:
        # if agent_live_episodes >= max_live_episodes:
        if (episode_number + 1) % max_live_episodes == 0:
            print("!!!!!!!!!!!!!!!!!!!SWAPPING LIVE AGENTS!!!!!!!!!!!!!!!!!!")
            # swap acitve and frozen networks and possibly update the max agent live episodes field
            active, frozen = swap_networks(active_network["training"], frozen_network["training"])
            active_network["training"] = active
            frozen_network["training"] = frozen
            active, frozen = swap_networks(active_network["target"], frozen_network["target"])
            active_network["target"] = active
            frozen_network["target"] = frozen
            print('active target epsilon: ', active_network["target"].epsilon)
            print('active training epsilon: ', active_network["training"].epsilon)
            print('frozen target epsilon: ', frozen_network["target"].epsilon)
            print('frozen training epsilon: ', frozen_network["training"].epsilon)
            print(active_network['training'].player, active_network['target'].player,
                  frozen_network['training'].player, frozen_network['target'].player)
            # exit()
            active_network['training'].current_training_episodes = 0
            active_network['target'].current_training_episodes = 0
            agent_live_episodes = 0

#todo: make sure that the fix you added to this is actually a fix
def swap_networks(network1, network2):
    print("A")
    # out1 = open("temp1_weights.pickle", "wb")
    # pickle.dump(network1.model, out1)
    # out1.close()

    # network1.model.save("net1_weights.h5")
    # network2.model.save('')
    network1.model = network2.model
    print("B")
    temp1 = DQNAgent(network2.currentState, network2.player)
    temp1.memory = copy.deepcopy(network2.memory)
    temp1.currentState = copy.deepcopy(network2.currentState)
    # temp1.model = network2.model
    print("C")
    # in1 = open("temp1_weights.pickle", "rb")
    # in1.close()
    print("D")
    temp1.epsilon = network2.epsilon
    temp1.current_training_episodes = network2.current_training_episodes
    temp1.max_training_episodes = network2.max_training_episodes
    temp1.max_agent_live_episodes = network2.max_agent_live_episodes
    temp1.player = network2.player

    # out2 = open("temp2_weights.pickle", "wb")
    # network2.model.save("p2_weights.h5")
    # pickle.dump(network2.model, out2)
    # out2.close()
    temp2 = DQNAgent(network1.currentState, network1.player)
    temp2.memory = copy.deepcopy(network1.memory)
    temp2.currentState = copy.deepcopy(network1.currentState)
    temp2.model = network1.model
    temp1.model = network2.model

    # in2 = open("temp2_weights.pickle", "rb")
    # temp2.model = models.load_model('net1_weights.h5')
    # temp1.model = models.load_model('p2_weights.h5')
    # in2.close()
    temp2.epsilon = network1.epsilon
    temp2.current_training_episodes = network1.current_training_episodes
    temp2.max_training_episodes = network1.max_training_episodes
    temp2.max_agent_live_episodes = network1.max_agent_live_episodes
    temp2.player = network1.player
    # temp2.model = network1.model

    # network1 = temp2
    # network2 = temp1
    print(network1.epsilon)
    print(network2.epsilon)
    return temp1, temp2


def play_checkers():
    pass

def main():
    # live_ranges = np.arange(10, 17)
    live_ranges = np.array([10, 11, 12])
    live_ranges = 2 ** live_ranges
    for live_range in live_ranges:
        maxTrainingExp = int(math.log2(live_range))
        #reminder remember to change to 6 as min bound later
        #todo: I wonder if the way you are removing stuff from an array is inefficient
        for trainingExp in range(8, maxTrainingExp):
            training_range = 2 ** trainingExp
            for mem_cap_exp in range(9,10):
                for epsilon_decay in range(5,10, 2):
                    max_memory_capacity = 2 ** mem_cap_exp

                    # epsilon_decay_rate = .9 + (epsilon_decay / 100)
                    epsilon_decay_rate = .999 + epsilon_decay / 10000
                    print("TRAINING AGENT WITH PARAMETERS: ")
                    print(live_range, training_range, max_memory_capacity, epsilon_decay_rate)
                    #train_model(live_range, training_range, max_memory_capacity, epsilon_decay_rate)
                    # epsilon_decay_rate = (epsilon_decay_rate / 10) + .9
                    train_model(live_range, training_range, max_memory_capacity, epsilon_decay_rate)
                    exit()
if __name__ == "__main__":
    main()