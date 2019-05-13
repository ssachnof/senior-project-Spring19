import unittest
import numpy as np
from source import makeMove
from source import constants
from source import driver
class TestCases(unittest.TestCase):

    # no piece is jumped over
    def test1_is_valid_jump(self):
        initial_board = get_testing_array([(3, 4)], [- 1])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (1, 1), True))
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (5, 2), False))
        initial_board = get_testing_array([(3, 4)], [1])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (4, 5), True))
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (2, 3), False))

    # current player's piece is jumped over
    def test2_is_valid_jump(self):
        # bottom left jump
        initial_board = get_testing_array([(3, 4), (4, 3)], [- 1, -2])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (5, 2), True))

        # top right jump
        initial_board = get_testing_array([(3, 4), (2, 5)], [- 1, -200])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), True))

    # a piece is located at the desired landing location
    def test3_is_valid_jump(self):
        initial_board = get_testing_array([(3, 4), (2, 5), (1, 6)], [- 1, 2, -3])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), True))

        initial_board = get_testing_array([(3, 4), (2, 5), (1, 6)], [- 1, 2, 3])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), True))

        initial_board = get_testing_array([(3, 4), (2, 5), (1, 6)], [1, -2, 3])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), True))

        initial_board = get_testing_array([(3, 4), (2, 5), (1, 6)], [1, -2, -3])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), True))




    # a backwards jump is made for a non-king
    def test4_is_valid_jump(self):
        initial_board = get_testing_array([(3, 4), (2, 5)], [- 100, 2])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), False))

        initial_board = get_testing_array([(3, 4), (4, 5)], [100, -2])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertFalse(makeMove.is_valid_jump(initial_state, (3, 4), (5, 6), False))

    # legal jumps in all 4 directions for player 1
    def test5_is_valid_jump(self):
        initial_board = get_testing_array([(3, 4), (2, 5)], [100, -2])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), False))

        initial_board = get_testing_array([(3, 4), (4, 5)], [100, -2])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (5, 6), True))

        initial_board = get_testing_array([(3, 4), (2, 3)], [100, -2])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (1, 2), False))

        initial_board = get_testing_array([(3, 4), (4, 3)], [100, -2])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (5, 2), True))


    # legal jumps in all 4 directions for player 2
    def test6_is_valid_jump(self):
        initial_board = get_testing_array([(3, 4), (2, 5)], [-100, 2])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (1, 6), True))

        initial_board = get_testing_array([(3, 4), (4, 5)], [-100, 2])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (5, 6), False))

        initial_board = get_testing_array([(3, 4), (2, 3)], [-100, 2])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (1, 2), True))

        initial_board = get_testing_array([(3, 4), (4, 3)], [-100, 2])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertTrue(makeMove.is_valid_jump(initial_state, (3, 4), (5, 2), False))


    # check backwards move directions for player 2
    def test1_is_backwards_move(self):
        initial_board = get_testing_array([(3, 4)], [- 1])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (2, 2)))
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (2, 5)))
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (1, 1)))
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (1, 6)))

    # check backwards move directions for player 1
    def test2_is_backwards_move(self):
        initial_board = get_testing_array([(3, 4)], [1])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (4, 5)))
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (4, 3)))
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (5, 6)))
        self.assertTrue(makeMove.is_backwards_move(initial_state, (3, 4), (5, 2)))

    # check non-backwards move directions for player 2
    def test3_is_backwards_move(self):
        initial_board = get_testing_array([(3, 4)], [- 1])
        initial_state = driver.State(initial_board, constants.PLAYER2)
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (4, 5)))
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (4, 3)))
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (5, 6)))
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (5, 2)))

    # check non-backwards move directions for player 1
    def test4_is_backwards_move(self):
        initial_board = get_testing_array([(3, 4)], [1])
        initial_state = driver.State(initial_board, constants.PLAYER1)
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (2, 3)))
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (2, 5)))
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (1, 2)))
        self.assertFalse(makeMove.is_backwards_move(initial_state, (3, 4), (1, 6)))



    # check legal and illegal backwards moves
    def test1_is_valid_move(self):
        pass

    # check moves that are OOB
    def test2_is_valid_move(self):
        pass

    # check piece dne on board case
    def test3_is_valid_move(self):
        pass

    # check illegal jump cases
    def test4_is_valid_move(self):
        pass

    # check 1 legal move of each of the following types: jump(king, nonking), backwards move, normal move
    def test5_is_valid_move(self):
        pass






######## testing helper functions #########

# piece_locations: list of tuples representing where pieces should be located on the board
# piece_values:
def get_testing_array(piece_locations, piece_values):
    arr = np.zeros(shape=(8,8), dtype=int)
    for i in range(len(piece_locations)):
        location = piece_locations[i]
        row = location[0]
        col = location[1]
        arr[row, col] = piece_values[i]
    return arr
