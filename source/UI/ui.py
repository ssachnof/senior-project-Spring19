import pygame
from source import driver
from source import constants
from source.Controller import DQNController
from tensorflow.keras import models
import math


WIDTH = 800
HEIGHT = 800
GREEN    = (   0,   102,   0)
BLACK = (      0,    0,    0)
RED = (        204,  0,    0)
STARTING_PLAYER = constants.PLAYER1


class Game:
    def __init__(self, staring_player, networkFileName):
        # backend state record
        self.backend_state = driver.get_initial_state(staring_player)
        self.network = models.load_model(networkFileName)

        # frontend state record
        game_state = DQNController.map_backend_state(self.backend_state)
        self.board = game_state[1]
        self.playerTurn = game_state[0]


    def draw(self, surface):
        # draw a bunch of verticle lines
        y = 0
        x = 0

        # draw the horrizontal/ vertical lines of the board
        for i in range(9):
            pygame.draw.line(surface, BLACK, (x, y), (x, y + 800))
            x += 100
        x = 0
        for i in range(9):
            pygame.draw.line(surface, BLACK, (x, y), (x + 800, y))
            y += 100

        for r in range(8):
            for c in range(8):
                x1 = 100 * c
                x2 = x1 + 100

                y1 = 100 * r
                y2 = y1 + 100


                center = ((x1 + x2)//2, (y1 + y2)//2)
                # Ω(self.board)
                if self.board[r][c] in ['x', 'X']:
                    pygame.draw.circle(surface, RED, center, 30)
                    if self.board[r][c] == 'X':
                        pygame.draw.circle(surface, BLACK, center, 15)
                elif self.board[r][c] in ['o', 'O']:
                    pygame.draw.circle(surface, BLACK, center, 30)
                    if self.board[r][c] == 'O':
                        pygame.draw.circle(surface, RED, center, 15)

    def play(self, start_pos, end_pos):
        r, c= 0, 1
        r1, c1 = start_pos
        r2, c2 = end_pos
        # map the frontend
        diff = math.sqrt(((r2 - r1) **2) + ((c2 - c1) ** 2))
        if diff > 1.5:
            med_row = (r1 + r2) // 2
            med_col = (c1 + c2) // 2
            self.board[med_row][med_col] = '_'
            self.backend_state.board[med_row][med_col] = 0
        temp = self.board[r1][c1]
        self.board[r1][c1] = '_'
        self.board[r2][c2] = temp
        if r2 == 0:
            self.board[r2][c2] = 'X'
            temp = self.backend_state.board[r1][c1]
            self.backend_state.board[r1][c1] = self.backend_state.board[r2][c2]
            self.backend_state.board[r2][c2] = temp * 100
        else:

            # map the backend
            temp = self.backend_state.board[r1][c1]
            self.backend_state.board[r1][c1] = self.backend_state.board[r2][c2]
            self.backend_state.board[r2][c2] = temp
        if self.is_complete():
            exit("USER Player won!!!!!")

        self.playerTurn *= -1
        self.backend_state.playerTurn = self.playerTurn
        action = DQNController.get_best_move(self.backend_state, self.network)
        new_state = DQNController.move_agent_player(self.backend_state, action)


        self.backend_state = new_state
        game_state = DQNController.map_backend_state(self.backend_state)
        self.board = game_state[1]
        self.playerTurn = game_state[0]

        # print(self.board)

    # def is_complete(self):
    #     p1_pieces, p2_pieces = [], []
    #     for r in self.board:
    #         for c in r:

    def is_legal_move(self, start_pos, end_pos):
        r1, c1 = start_pos
        r2, c2 = end_pos
        # map the frontend
        # print(self.board[r2][c2])
        # exit()
        diff = math.sqrt(((r2 - r1) **2) + ((c2 - c1) ** 2))
        if self.board[r1][c1] not in ['x', 'X']:
            return False
        elif self.board[r2][c2] != '_':
            # print("running!!!!!")
            return False
        elif diff > 1.5:
            med_row = (r1 + r2) // 2
            med_col = (c1 + c2) // 2
            if self.board[med_row][med_col] not in ['o', 'O']:
                return False
        elif r2 > r1 and self.board[r1][c1] != 'X':
            # print("running!!!!")
            return False
        elif abs(r1 - r2) != abs(c1 - c2):
            # print("Running!!!")
            return False
        return True
    def map_mouse_click(self, pos):
        '''

        :param pos: tuple(int, int) representing the location of where the mouse clicked
        :return: tuple(int, int)

        returns the center of the square that's closest to pos
        '''


        c = pos[0] // 100
        r = pos[1] // 100

        return r,c

    def is_complete(self):
        p1_pieces, p2_pieces = [], []
        for r in self.board:
            for c in r:
                if c in ['x', "X"]:
                    p1_pieces.append(c)
                elif c != '_':#this case probably isn't necessary, since termination returned in the backend for the agent
                    p2_pieces.append(c)
        print(p1_pieces)
        print(p2_pieces)
        return not ((len(p1_pieces) > 0) and (len(p2_pieces) > 0))

def main():
    pygame.init()
    size = (WIDTH, HEIGHT)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    clock.tick(60)
    game_board = pygame.image.load('Images/icon.jpg')
    pygame.display.set_icon(game_board)
    game = Game(constants.PLAYER1, '../p2_weights.h5')
    loc_start, loc_end = None, None


    while True:
        screen.fill(GREEN)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if loc_start is None:
                    loc_start = pygame.mouse.get_pos()
                elif loc_end is None:
                    loc_end = pygame.mouse.get_pos()
                    loc_start = game.map_mouse_click(loc_start)
                    loc_end = game.map_mouse_click(loc_end)
                    if game.is_complete():
                        exit("Somebody won!!!!")
                    if game.is_legal_move(loc_start, loc_end):
                        game.play(loc_start, loc_end)
                    loc_start, loc_end = None, None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z and event.mod == pygame.KMOD_LMETA:
                    loc_start, loc_end = None, None

            game.draw(screen)
            pygame.display.update()



if __name__ == "__main__":
    main()
