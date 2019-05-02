'''
contains the logic for making a move
'''

import math


'''
represents a State in the environment
board: list[list[piece]], isTerminal: boolean, playerTurn: int
'''
class State:
    def __init__(self, board, playerTurn):
        self.board = board
        self.playerTurn = playerTurn




'''
a move is valid
'''
def is_valid_move(initial_board, initial_piece_location, final_piece_location, isKing):
    is_jump = False

    # piece dne case
    if initial_piece_location is None:
        return False

    # determine if the move is a jump move type
    move_difference = (initial_piece_location[0] - final_piece_location[0],
                       initial_piece_location[1] - final_piece_location[1])
    if math.sqrt((move_difference[0] ** 2) + (move_difference[1] ** 2)) > math.sqrt(2):
        is_jump = True

    #check OOB cases
    if (final_piece_location[0] not in list(range(8))) or (final_piece_location[1] not in list(range(8))):
        return False

    # non-king backwards move case
    elif is_backwards_move(initial_board, initial_piece_location, final_piece_location) and not isKing:
        return False

    # jump cases: jump onto a piece or jump without jumping over a piece
    elif is_jump and not is_valid_jump(initial_board, initial_piece_location, final_piece_location, isKing):
        return False

    return True





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
            if initial_board[row][col] == piece_num:
                piece_initial_location = (row, col)
            elif initial_board[row][col] // 100 == piece_num:
                piece_initial_location = (row, col)
                isKing = True
    piece_final_location = get_final_piece_location(initial_board, piece_initial_location, move_num)

    # check all other cases for valid moves
    if not is_valid_move(initial_board, piece_initial_location, piece_final_location, isKing):
        return {"done": True, "initial_state": initial_state, "final_state": initial_state, "reward": -2}

    # alter the state if the move that was made was valid
    done, final_state, reward = make_move(initial_state, piece_initial_location, piece_final_location)
    return {"done" : done, "initial_state": initial_state, "final_state": final_state, "reward": reward}

    # todo: think about how to map move num 96 tonight after class!-- probably should handle this case separately
    #  in training loop





