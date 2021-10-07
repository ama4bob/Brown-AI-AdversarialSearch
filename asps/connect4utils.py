import numpy as np
import copy


def create_board(shape=(6, 7)):
    return np.zeros(shape, dtype=int)


def drop_piece(board, row, col, piece):
    board = copy.deepcopy(board)
    board[row][col] = piece
    return board


def is_valid_location(board, col):
    return board[-1][col] == 0


def get_next_open_row(board, col):
    rows = board.shape[0]
    for r in range(rows):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flipud(board))


def all_connect_four_slices(board):
    rows, cols = board.shape
    connect_fours = []
    # All horizontal four-in-a-rows
    for c in range(cols - 3):
        connect_fours.append(board[:, c : c + 4])
    # All vertical four-in-a-rows
    for r in range(rows - 3):
        connect_fours.append(board[r : r + 4, :].T)
    # All diagonal four-in-a-rows
    for r in range(rows - 3):
        for c in range(cols - 3):
            # Add both diagonals for each 4x4 square in board
            square = board[r : r + 4, c : c + 4]
            connect_fours.append(
                [
                    square.diagonal(),
                    np.fliplr(square).diagonal(),
                ]
            )
    return np.concatenate(connect_fours).astype(int)


def winning_move(board, piece):
    return (all_connect_four_slices(board) == piece).all(axis=1).any()
