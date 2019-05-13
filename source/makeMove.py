'''
contains the logic for making a move,

note that for at least the first prototype you will be ignoring the possibility of double jumps
(ie. only take 1 jump if 2+ continuous jumps can be made). Unless you somehow make an infinite action
space, this will be difficult to implement
'''


import math
import numpy as np
from source import constants


'''
represents a State in the environment
board: list[list[piece]], isTerminal: boolean, playerTurn: int
'''
class State:
    def __init__(self, board, playerTurn):
        self.board = board
        self.playerTurn = playerTurn

    # returns the representation of the current state in 1d(ie. an array of size 65) form
    # this is needed in order to properly pass a state into the neural network
    def flatten(self):
        return np.hstack([self.board.reshape((64, )), np.array([self.playerTurn])])




'''
a move is valid
'''


# todo: add case to ensure that the desired space is vacant
def is_valid_move(initial_state, initial_piece_location, final_piece_location, isKing):
    # piece dne case
    if initial_piece_location is None:
        return False

    #check OOB cases
    if (final_piece_location[0] not in list(range(8))) or (final_piece_location[1] not in list(range(8))):
        return False

    # ensure the destination square is unoccupied
    elif initial_state.board[final_piece_location[0], final_piece_location[1]] != 0:
        return False

    # non-king backwards move case
    elif is_backwards_move(initial_state, initial_piece_location, final_piece_location) and not isKing:
        return False

    # jump cases: jump onto a piece or jump without jumping over a piece
    elif is_jump(initial_piece_location, final_piece_location) \
            and not is_valid_jump(initial_state, initial_piece_location, final_piece_location, isKing):
        return False

    return True


'''
returns true if the move that was made was a jump
'''
def is_jump(initial_piece_location, final_piece_location):
    move_difference = (initial_piece_location[0] - final_piece_location[0],
                       initial_piece_location[1] - final_piece_location[1])
    return math.sqrt((move_difference[0] ** 2) + (move_difference[1] ** 2)) > math.sqrt(2)



'''
for a jump to be valid:
1) there must be an opponent's piece in the diagonal path
2) there must not be a piece in the final destination for the piece
'''
def is_valid_jump(initial_state, initial_piece_location, final_piece_location, isKing):
    # determine if there is a piece in the diagonal path-- a opp piece must exist at the midpoint between
    # the initial location and the final location

    mid = ((initial_piece_location[0] + final_piece_location[0]) // 2,
           (initial_piece_location[1] + final_piece_location[1]) // 2)


    # opp piece must exist at mid
    if initial_state.board[mid[0], mid[1]] == 0 or \
            (initial_state.board[mid[0], mid[1]] // abs(initial_state.board[mid[0], mid[1]]) != initial_state.playerTurn * -1):
        return False

    # ensure no piece in landing location
    elif initial_state.board[final_piece_location[0], final_piece_location[1]] != 0:
        return False

    # ensure that no backwards move has been made for a non-king
    elif (not isKing) and is_backwards_move(initial_state, initial_piece_location, final_piece_location):
        return False

    return True





def is_backwards_move(initial_state, initial_piece_location, final_piece_location):
    initial_board = initial_state.board.copy()

    if initial_state.playerTurn == constants.PLAYER1:
        return final_piece_location[0] - initial_piece_location[0] > 0
    else:
        return final_piece_location[0] - initial_piece_location[0] < 0



'''
returns a memory tuple of (done: boolean, initial_state: state, final_state: state, reward)
initial_state: state, action: int

king pieces will be represented by piece number * 100

win : 1
loss : -1
illegal move : -2
tie : 0
else : 0
'''



def get_next_state(initial_state, action):

    initial_board, final_board = initial_state.board
    isKing = False
    piece_initial_location = None

    # map action to piece numbers and move numbers
    move_num = action % 8
    piece_num = (action // 8) + 1
    piece_num *= initial_state.playerTurn
    # find piece_num on the board
    for row in range(len(initial_board)):
        for col in range(len(initial_board[0])):
            if initial_board[row, col] == piece_num:
                piece_initial_location = (row, col)
            elif initial_board[row, col] // 100 == piece_num:
                piece_initial_location = (row, col)
                isKing = True
    # note that because the move mapping of actually occurs inside this, it might be better to just pass in the
    # state, idk
    piece_final_location = get_final_piece_location(initial_state, piece_initial_location, move_num)

    # check all other cases for valid moves
    if not is_valid_move(initial_state, piece_initial_location, piece_final_location, isKing):
        return True, initial_state, action, initial_state, -2

    # alter the state if the move that was made was valid
    done, final_state, reward = make_move(initial_state, piece_initial_location, piece_final_location)
    return done, initial_state, action, final_state, reward



# note that this functions assumes a valid move was made
# returns a valid move for a piece

# this essentially does the mapping for the move from an integer move number value to a final move location
def get_final_piece_location(initial_state, piece_initial_location, move_num):
    row = 0
    col = 1
    initial_loc_change = np.array([0, 0])
    player_turn = initial_state.playerTurn
    # map the moves according to player 1's offsets
    if move_num in [constants.BACKWARD_LEFT, constants.BACKWARD_JUMP_LEFT]:
        initial_loc_change[row] += 1
        initial_loc_change[col] -= 1

    elif move_num in [constants.BACKWARD_RIGHT, constants.BACKWARD_JUMP_RIGHT]:
        initial_loc_change[row] += 1
        initial_loc_change[col] += 1

    elif move_num in [constants.FORWARD_LEFT, constants.FORWARD_JUMP_LEFT]:
        initial_loc_change[row] -= 1
        initial_loc_change[col] -= 1
    # either forward right or forward jump right
    else:
        initial_loc_change[row] -= 1
        initial_loc_change[col] += 1

    # double change distance if a jump
    if move_num > 3:
        initial_loc_change *= 2

    # negate if player 2
    if player_turn == constants.PLAYER2:
        initial_loc_change *= -1

    return (piece_initial_location[row] + initial_loc_change[row],
            piece_initial_location[1] + initial_loc_change[col])


'''
returns done, final_state, reward
assumes the given final location is valid
'''


def make_move(initial_state, piece_initial_location, piece_final_location):
    final_board = initial_state.board.copy()
    if is_jump(piece_initial_location, piece_final_location):
        ...

    # swap the piece initial location and piece final location
    temp = final_board[piece_initial_location[0], piece_initial_location[1]]
    final_board[piece_initial_location[0], piece_initial_location[1]] = 0
    final_board[piece_final_location[0], piece_final_location[1]] = temp

    final_state = State(final_board, initial_state.playerTurn * -1)
    done, reward = get_reward(final_state)

    return done, final_state, reward



# returns the done, reward for being in a given state
def get_reward(final_state):
    next_player_pieces = []
    for row_index in range(len(final_state.board)):
        for col_index in range(len(final_state.board[0])):
            board_val = final_state.board[row_index, col_index]
            if board_val != 0 and abs(board_val) // board_val == final_state.playerTurn:
                next_player_pieces.append((row_index, col_index))

    if next_player_pieces == []: # somebody won the game
        return True, 1

    # see if there exists a valid move for at least 1 piece of the next player-- check for tie condition
    for piece_loc in next_player_pieces:
        piece_val = final_state.board[piece_loc[0], piece_loc[1]]
        isKing = False
        if abs(piece_loc) >= 100:
            isKing = True
            for move_num in range(8):
                if is_valid_move(final_state, piece_loc,
                                 get_final_piece_location(final_state, piece_loc, move_num), isKing):
                    # the game is not terminal/ not a tie
                    return False, 0
    # the game resulted in a tie
    return True, 0

