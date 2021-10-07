# authors: Jason Crowley and Eli Zucker

###############################################################################
# State and Action Representations:
#
# - A (game) state is a Connect4State that contains game board and the index
#   of the player to move
#
# - An action is a 0-indexed column in which a player would like to drop their piece
#
# - See the main function at the bottom of this file for an example.
#
###############################################################################


from typing import Tuple
from adversarialsearchproblem import AdversarialSearchProblem, GameUI, GameState
from . import connect4utils as c4utils
import numpy as np
import pygame
import sys


class Connect4State(GameState):
    def __init__(self, board, ptm):
        """
        Inputs:
                board - represented as a 2D NumPy array of integers.
                Each integer in the board is a 0, 1, or 2, which indicates
                that the corresponding cell of the Connect Four board is empty,
                occupied by player 1, or occupied by player 2, respectively.

                ptm - the index of the player to move,
                where 0 corresponds to player 1, who moves first,
                and 1 corresponds to player 2, who moves second.
        """
        self.board = board
        self.ptm = ptm

    def player_to_move(self):
        return self.ptm


# In Connect 4, an action consists of placing a piece in a particular column. As such, the type of
# actions for Connect4Problem is an int, which simply specifies which column to drop the current
# piece into.
Action = int


class Connect4Problem(AdversarialSearchProblem[Connect4State, Action]):
    DEFAULT_ROWS = 6
    DEFAULT_COLS = 7

    def __init__(self, dims=(DEFAULT_ROWS, DEFAULT_COLS), board=None, player_to_move=0):
        """
        Inputs:
                dims - dimensions of the board, represented by a tuple: (<# of rows>, <# of columns>)
                board - 2D NumPy array of integers (as in Connect4State)
                player_to_move - index of the player to move (as in Connect4State).

                The board and player_to_move together constitute the start state
                of the game
        """
        if board == None:
            board = c4utils.create_board(dims)
        self._rows, self._cols = board.shape
        self._start_state = Connect4State(board, player_to_move)

    def heuristic_func(self, state: Connect4State, player_index):
        player_index += 1
        board = state.board
        cols = board.shape[1]
        score = 0

        # Score center column
        center_array = list(board[:, cols // 2])
        center_count = center_array.count(player_index)
        score += center_count * 3

        # Score all possible connect 4 "slices"
        for slice in c4utils.all_connect_four_slices(board):
            score += Connect4Problem.evaluate_slice(list(slice), player_index)

        return score

    def get_available_actions(self, state):
        actions = {
            col
            for col in range(self._cols)
            if c4utils.is_valid_location(state.board, col)
        }
        return actions

    def transition(self, state, action):
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)

        row = c4utils.get_next_open_row(state.board, action)
        board = c4utils.drop_piece(state.board, row, action, state.ptm + 1)
        return Connect4State(board, 1 - state.ptm)

    def is_terminal_state(self, state):
        p1_wins = c4utils.winning_move(state.board, 1)
        p2_wins = c4utils.winning_move(state.board, 2)
        assert not (p1_wins and p2_wins)
        return p1_wins or p2_wins or not self.get_available_actions(state)

    def evaluate_terminal(self, state):
        assert self.is_terminal_state(state)

        if c4utils.winning_move(state.board, 1):
            return [float("inf"), float("-inf")]
        elif c4utils.winning_move(state.board, 2):
            return [float("-inf"), float("inf")]
        else:
            return [0, 0]

    @staticmethod
    def evaluate_slice(slice, player_index):
        score = 0
        opp_index = player_index % 2 + 1

        if slice.count(player_index) == 4:
            score += 100
        elif slice.count(player_index) == 3 and slice.count(0) == 1:
            score += 5
        elif slice.count(player_index) == 2 and slice.count(0) == 2:
            score += 2

        if slice.count(opp_index) == 3 and slice.count(0) == 1:
            score -= 4

        return score

    @staticmethod
    def visualize_state(state):
        c4utils.print_board(state.board)
        print()


class Connect4GUI(GameUI):
    # Define GUI colors
    BOARD = (0, 0, 255)  # blue
    EMPTY = (0, 0, 0)  # black
    P1 = (255, 0, 0)  # red
    P2 = (255, 255, 0)  # yellow
    COLOR_MAP = [EMPTY, P1, P2]

    def __init__(self, asp: Connect4Problem, squaresize=100):
        self._asp = asp
        self._state = asp.get_start_state()
        self._rows, self._cols = asp._start_state.board.shape
        self._cursor = self._cols // 2

        self.squaresize = int(squaresize)
        self.width = self._cols * self.squaresize
        self.height = (self._rows + 1) * self.squaresize
        self.radius = self.squaresize // 2 - 5

        # Initial pygame window setup:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.draw_board()
        self.draw_cursor()
        self.render()

    def render(self):
        pygame.display.update()

    # Process each event in the GUI window.
    # Required for the GUI to function as a standard OS application window.
    def process_window_event(self, event):
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self._cursor -= 1
            if event.key == pygame.K_RIGHT:
                self._cursor += 1
            self._cursor = np.clip(self._cursor, 0, self._cols - 1)

            self.draw_cursor()
            self.render()

    # Continuously process events until action is found, then returns that action.
    def get_user_input_action(self):
        user_action = None
        available_actions = self._asp.get_available_actions(self._state)

        while user_action not in available_actions:
            for event in pygame.event.get():
                self.process_window_event(event)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    user_action = self._cursor

        return user_action

    # Update the board state being rendered.
    def update_state(self, state: Connect4State):
        self._state = state
        self.draw_board()
        self.draw_cursor()

    def draw_piece(self, row, col, piece):
        pygame.draw.circle(
            self.screen,
            Connect4GUI.COLOR_MAP[piece],
            (int((col + 0.5) * self.squaresize), int((row + 1.5) * self.squaresize)),
            self.radius,
        )

    def draw_cursor(self):
        pygame.draw.rect(
            self.screen, Connect4GUI.EMPTY, (0, 0, self.width, self.squaresize)
        )
        piece = self._state.player_to_move() + 1
        self.draw_piece(-1, self._cursor, piece)

    def draw_board(self):
        # Fill in board area:
        pygame.draw.rect(
            self.screen,
            Connect4GUI.BOARD,
            (0, self.squaresize, self.width, self.height),
        )
        # Fill in circles:
        board = self._state.board
        for (r, c), piece in np.ndenumerate(np.flipud(board)):
            self.draw_piece(r, c, piece)


def main():
    """
    Provides an example of the Connect4Problem class being used.
    """
    t = Connect4Problem()

    # Start with an empty board with player 2 to move
    s0 = Connect4State(c4utils.create_board(), 1)
    # Player 2 drops a piece in the center column
    s1 = t.transition(s0, 3)
    # Player 1 also drops a piece in the center column
    s2 = t.transition(s1, 3)

    # Reproduce what should have happened in the states above:
    # Start with an empty board
    board = c4utils.create_board()
    assert (s0.board == board).all()
    assert s0.ptm == 1
    # Player 2 drops a piece at (0, 3)
    board[0][3] = 2
    assert (s1.board == board).all()
    assert s1.ptm == 0
    # Player 1 drops a piece at (1, 3)
    board[1][3] = 1
    assert (s2.board == board).all()
    assert s2.ptm == 1

    winning_board = c4utils.create_board()
    winning_board[0][0:4] = 1
    winning_board[1][0:3] = 2
    winning_state = Connect4State(winning_board, 1)

    assert t.is_terminal_state(winning_state)
    assert t.evaluate_terminal(winning_state) == [float("inf"), float("-inf")]

    tie_board = c4utils.create_board()
    tie_board[:, 0::2] = np.array([1, 1, 1, 2, 2, 2])[:, None]
    tie_board[:, 1::2] = np.array([2, 2, 2, 1, 1, 1])[:, None]
    tie_state = Connect4State(tie_board, 0)

    assert t.is_terminal_state(tie_state)
    assert t.evaluate_terminal(tie_state) == [0, 0]

    print("State 0:")
    Connect4Problem.visualize_state(s0)

    print("State 1:")
    Connect4Problem.visualize_state(s1)

    print("State 2:")
    Connect4Problem.visualize_state(s2)

    print("Winning State:")
    Connect4Problem.visualize_state(winning_state)

    print("Tie State:")
    Connect4Problem.visualize_state(tie_state)


if __name__ == "__main__":
    main()
