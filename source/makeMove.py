'''
this file handles the making of a move, and returns the result back to the agent
'''




'''
represents a piece on the board
player is an integer representing the player who owns the piece
isKing is a boolean which is true if the piece is a king
board_loc represents the location the piece is on the board


player: int, isKing: bool, board_loc: tuple[row: int, col: int]
'''

class Piece:
    def __init__(self, player, isKing, board_loc):
        self.player = player
        self.isKing = isKing
        self.board_loc = board_loc


'''
represents a move for a given piece on the board
piece: piece, new_loc: tuple[row: int, col: int], move_type: int
'''
class Move:
    def __init__(self, piece, new_loc, move_type):
        self.reward = None
        self.piece = piece
        self.new_loc = new_loc
        self.move_type = move_type


'''
represents a State in the environment
board: list[list[piece]], isTerminal: boolean, playerTurn: int
'''
class State:
    def __init__(self, board, isTerminal, playerTurn):
        self.board = board
        self.isTerminal = isTerminal
        self.playerTurn = playerTurn




'''
determines if Move move is valid given a state
'''
def is_valid_move(state, move):
    pass



'''
returns the reward of a given state
'''
def get_reward(state, final_state):
    pass
