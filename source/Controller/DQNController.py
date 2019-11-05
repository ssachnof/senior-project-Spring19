from DQNAgent import *
from source import makeMove
from source import constants


def get_best_move(current_state, network):
    # networkFileName is the .h5 file that contains the neurual network weights for the agent
    # note that current state is a backend state representation


    lm = list(makeMove.get_all_legal_moves(current_state))
    allActions = network.predict(current_state.flatten())[0]
    q_values = []
    for l in lm:
        # print(lm_i)
        q_values.append(allActions[l])
    # print(q_values)
    # print(lm)
    # exit()
    max_q_pos = np.argmax(np.array(q_values))
    return lm[max_q_pos]


def map_backend_state(current_state):
    '''
    converts the state representation from the backend to the state representation of the frontend
    this is so it is easier to draw on screen(and is understandable by the user)

    p1 normal piece: x
    p1 King: X

    p2 normal piece: o
    p2 King: O

    unoccupied space: _


    :param current_state[makeMove.state]
    :return: tuple(current_turn, a 2d array of chars representing the board)
    '''
    board = []
    for r in range(8):
        row = []
        for c in range(8):
            piece_val = current_state.board[r][c]
            if piece_val == 0:
                row.append('_')
            elif abs(piece_val) / piece_val == constants.PLAYER1:
                if abs(piece_val) >= 100:
                    row.append('X')
                else:
                    row.append('x')
            else:
                if abs(piece_val) >= 100:
                    # p2 piece
                    row.append('O')
                else:
                    row.append('o')

        board.append(row)
    return current_state.playerTurn, board




def is_valid_move(action):
    '''
    policy ensuring that a move made by the agent is valid
    :return:
    '''
    pass



def get_new_board(state, action):
    pass


def move_agent_player(current_state, action):
    backend_state = makeMove.get_next_state(current_state, None, None, None, action, None)[3]
    print(backend_state.board)
    # print(backend_state.board)
    return backend_state


