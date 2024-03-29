'''
contains the logic for making a move,

note that for at least the first prototype you will be ignoring the possibility of double jumps
(ie. only take 1 jump if 2+ continuous jumps can be made). Unless you somehow make an infinite action
space, this will be difficult to implement
'''


import math
import numpy as np
import constants


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
        return np.array([np.hstack([self.board.reshape((64, )), np.array([self.playerTurn])])])




'''
a move is valid
'''


# todo: add case to ensure that the desired space is vacant
def is_valid_move(initial_state, initial_piece_location, final_piece_location, isKing):
    # piece dne case
    if initial_piece_location is None:
        # print("null initial location")
        return False

    #check OOB cases
    if (final_piece_location[0] not in list(range(8))) or (final_piece_location[1] not in list(range(8))):
        # print("invalid range")
        return False

    # ensure the destination square is unoccupied
    elif initial_state.board[final_piece_location[0], final_piece_location[1]] != 0:
        # print("piece loc is unoccupied")
        return False

    # non-king backwards move case
    elif is_backwards_move(initial_state, initial_piece_location, final_piece_location) and not isKing:
        # print("backwards move!!!!!")
        return False

    # jump cases: jump onto a piece or jump without jumping over a piece
    elif is_jump(initial_piece_location, final_piece_location) \
            and not is_valid_jump(initial_state, initial_piece_location, final_piece_location, isKing):
        # print("occupied jump move")
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


def checkBoard(initial_board, piece_num):
    for row_i in range(len(initial_board)):
        for col_i in range(len(initial_board[0])):
            if initial_board[row_i][col_i] == piece_num:
                return True
    return False


def get_next_state(initial_state, DQNAgent, maxMemorySize, distanceFromBest, action, legal_moves):
    legal_moves = get_all_legal_moves(initial_state)
    initial_board = initial_state.board
    # print('is: ', initial_state.board)
    # print('currentTurn: ', initial_state.playerTurn)
    isKing = False
    piece_initial_location = None
    # map action to piece numbers and move numbers
    move_num = action % 8
    piece_num = (action // 8) + 1
    piece_num *= initial_state.playerTurn
    # print(piece_num)
    # print(move_num)
    # find piece_num on the board
    for row in range(len(initial_board)):
        for col in range(len(initial_board[0])):
            if initial_board[row, col] == piece_num:
                piece_initial_location = (row, col)
            elif initial_board[row, col] // 100 == piece_num and not checkBoard(initial_board, piece_num):
                piece_initial_location = (row, col)
                isKing = True
    # print('ipl1: ', piece_initial_location, 'pn: ', piece_num, 'action: ', action, 'isKing: ', isKing)
    # note that because the move mapping of actually occurs inside this, it might be better to just pass in the
    # state, idk

    # if piece_initial_location is None:
    #     print("initial loc is none and will cause an exception")
    #     print("turn: ", initial_state.playerTurn)
    #     print("move num: ", move_num)
    #     print("piece num: ", piece_num)
    #     print(initial_state.board)
    #     exit()
    piece_final_location = get_final_piece_location(initial_state, piece_initial_location, move_num)

    # check all other cases for valid moves
    #todo: you should not even need this loop


    # while not is_valid_move(initial_state, piece_initial_location, piece_final_location, isKing):
    #     legal_moves = get_all_legal_moves(initial_state)
    #     # print(initial_state.board)
    #     action, distanceFromBest  = DQNAgent.get_next_action(maxMemorySize, legal_moves, distanceFromBest=distanceFromBest)
    #     isKing = False
    #     piece_initial_location = None
    #
    #     # map action to piece numbers and move numbers
    #     move_num = action % 8
    #     piece_num = (action // 8) + 1
    #     piece_num *= initial_state.playerTurn
    #     # print(piece_num)
    #     # print(move_num)
    #     # find piece_num on the board
    #     for row in range(8):
    #         for col in range(8):
    #             if initial_board[row, col] == piece_num:
    #                 piece_initial_location = (row, col)
    #             elif initial_board[row, col] // 100 == piece_num and not checkBoard(initial_board, piece_num):
    #                 piece_initial_location = (row, col)
    #                 isKing = True
    #     assert(piece_initial_location is not None)
    #     assert(move_num < 8)
    #     piece_final_location = get_final_piece_location(initial_state, piece_initial_location, move_num)
    #     if not is_valid_move(initial_state, piece_initial_location, piece_final_location, isKing):
    #         print('stuck!!!')
    #         print(initial_state.board)
    #         print(initial_state.board[piece_initial_location[0]][piece_initial_location[1]], piece_final_location)
    #         print(legal_moves)
    #         print(action)
    #         exit()
        # print(piece_initial_location, piece_final_location)
        # return True, initial_state, action, initial_state, -2
    # print("valid move found")
    # alter the state if the move that was made was valid
    #todo: after get_all_legal_moves is fixed, you should uncomment this!!!!!
    if not is_valid_move(initial_state, piece_initial_location, piece_final_location, isKing):
        # print('this should not run!!!!')
        # exit()
        print('ILLEGAL MOVE MADE BY AGENT!!!!!!!!')
        print(piece_initial_location, piece_final_location)
        return True, initial_state, action, initial_state, -2
    done, final_state, reward = make_move(initial_state, piece_initial_location, piece_final_location)
    # print('{{{{{{{{{{{{{{{')
    # print(final_state.board)
    # print('{{{{{{{{{{{{{')
    # print('action: ', action)
    # print('fs: ', final_state.board)
    # if debug:
    #     print("done: ", done)
    #     print("initial board: \n", initial_state.board)
    #     print("\n\nfinal board: \n", final_state.board)
    #     print("\n\nnext_turn ", final_state.playerTurn)
    #     print("reward: ", reward)
    #     exit("debug exit")
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
    # print("initial_loc: ", piece_initial_location)
    # print("change: ", initial_loc_change)
    # print("result: ", (piece_initial_location[row] + initial_loc_change[row],
    #        piece_initial_location[col] + initial_loc_change[col]))
    return (piece_initial_location[row] + initial_loc_change[row],
            piece_initial_location[col] + initial_loc_change[col])


'''
returns done, final_state, reward
assumes the given final location is valid
'''


def make_move(initial_state, piece_initial_location, piece_final_location):
    final_board = initial_state.board.copy()
    # you need to remove the piece that was jumped over
    x = (piece_final_location[0] - piece_initial_location[0]) **2
    y = (piece_final_location[1] - piece_initial_location[1]) **2
    fj = False
    if math.sqrt(x + y) > 1.5:
        # print('found_jump')
        # print('++++++++++++++++++')
        fj = True
        # print(initial_state.board)
    loop_running = False
    # unwanted mutation might be occuring in this function-- you are modifying initial state vs creating and modifying a final state
    if is_jump(piece_initial_location, piece_final_location):
        loc_change = np.array([piece_final_location[0] - piece_initial_location[0],
                      piece_final_location[1] - piece_initial_location[1]])
        loc_change = loc_change // 2
        loc_to_remove = (piece_initial_location[0] + loc_change[0], piece_initial_location[1] + loc_change[1])
        assert(0 != initial_state.board[loc_to_remove[0], loc_to_remove[1]])
        initial_state.board[loc_to_remove[0], loc_to_remove[1]] = 0
        loop_running = True
        final_board = initial_state.board.copy()
        # print('!!!!!!!!!!!!!!!')
        # print(initial_state.board)
        # print('!!!!!!!!!!!!!!!!')
        pass
    if fj and not loop_running:
        exit("JUMP LOOP DIDNT RUN!!!!!!")

    # swap the piece initial location and piece final location
    temp = final_board[piece_initial_location[0], piece_initial_location[1]]
    final_board[piece_initial_location[0], piece_initial_location[1]] = 0
    final_board[piece_final_location[0], piece_final_location[1]] = temp
    final_state = State(final_board, initial_state.playerTurn * -1)
    createKing(final_state, piece_final_location, initial_state.playerTurn)
    done, reward = get_reward(final_state)
    # if loop_running:
    #     print('+++++++++++')
    #     print(final_state.board)
    #     print('+++++++++++')
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
        if abs(piece_val) >= 100:
            isKing = True
        for move_num in range(8):
            if is_valid_move(final_state, piece_loc,
                             get_final_piece_location(final_state, piece_loc, move_num), isKing):
                # the game is not terminal/ not a tie
                return False, 0
    # the game resulted in a tie
    return True, 0


def createKing(final_state, pieceLoc, playerNum):
    row = 0
    pieceVal = final_state.board[pieceLoc[row], pieceLoc[1]]
    rowNum = pieceLoc[0]
    assert(final_state.board[pieceLoc[0], pieceLoc[1]] != 0)
    if abs(pieceVal) // 100 == 0:
        if rowNum == 0 and playerNum == constants.PLAYER1 and abs(pieceVal) <= 12:
            final_state.board[pieceLoc[row], pieceLoc[1]] *= 100
        elif rowNum == 7 and playerNum == constants.PLAYER2 and abs(pieceVal) <= 12:
            final_state.board[pieceLoc[row], pieceLoc[1]] *= 100

#todo: this is potentially returning out of bounds values
def get_all_legal_moves(currentState):
    '''

    :param currentState: state
    :return: dict[int: set[int]

    return value is in the following format: dict[pieceNum: legal_move_values]
    '''

    legal_moves = set()
    # for i in range(1, 13):
    #     legal_moves[i * currentState.playerTurn] = set()
    board = currentState.board
    for row_i in range(len(board)):
        for col_i in range(len(board[0])):
            board_value = board[row_i][col_i]
            if board_value != 0 and abs(board_value) // board_value == currentState.playerTurn:
                #then get all the legal moves for the current player's piece
                for move_num in range(8):
                    isKing = False
                    piece_index = abs(board_value)
                    if abs(board_value) >= 100:
                        isKing = True
                        piece_index = abs(board_value) // 100
                    initial_piece_loc = (row_i, col_i)
                    final_piece_location = get_final_piece_location(currentState, initial_piece_loc, move_num)
                    if is_valid_move(currentState, initial_piece_loc, final_piece_location, isKing):
                        # print('pn: ', piece_index, 'mn: ', move_num)
                        # print('arr_value: ', str(((abs(piece_index) - 1) * 8) + move_num))
                        # print('fpl: ', final_piece_location)
                        #legal_moves[piece_index] = legal_moves[piece_index] | {(move_num, piece_index)}

                        # this theoretically should not crash but still does
                        for i in range(96):
                            if (abs(piece_index) - 1) == (i // 8) and i % 8 == move_num:
                                arr_value = i
                                legal_moves = legal_moves | {arr_value}
                        # arr_value = ((abs(piece_index) - 1) * 8) + move_num
                        # legal_moves = legal_moves | {arr_value}#you may want to verify this later

    l = list(legal_moves)
    x = 1
    # print(legal_moves)
    return legal_moves


