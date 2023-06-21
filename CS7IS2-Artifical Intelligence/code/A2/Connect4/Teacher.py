import random
import numpy as np

def getValidColumns(board):
    columns = []
    for i in range(7):
        if board[0][i] == 0:
            columns.append(i)
    return columns


def set_move(state, column, player):
    for i in range(6):
        if state[i][column] != 0:
            state[i - 1][column] = player
            break
        elif i == 5:
            state[i][column] = player


def checkWin(state, player):
    for i in range(6):
        for j in range(4):
            if state[i][j] == state[i][j + 1] == state[i][j + 2] == state[i][j + 3] == player:
                return True

    for i in range(7):
        for j in range(3):
            if state[j][i] == state[j + 1][i] == state[j + 2][i] == state[j + 3][i] == player:
                return True

    for i in range(3):
        for j in range(4):
            if state[i][j] == state[i + 1][j + 1] == state[i + 2][j + 2] == state[i + 3][j + 3] == player:
                return True

    for i in range(3):
        for j in range(3, 7):
            if state[i][j] == state[i + 1][j - 1] == state[i + 2][j - 2] == state[i + 3][j - 3] == player:
                return True
    return False


class Teacher:
    def set_win(self, board, player=1):
        columns = getValidColumns(board)
        for column in columns:
            new_board = np.copy(board)
            set_move(new_board, column, player)
            if checkWin(board, player):
                return column

        return None

    def set_blockOpponentWin(self, board):
        return self.set_win(board, 2)

    def set_random(self, board):
        while True:
            col = random.randint(0, 6)
            if board[0][col] == 0:
                return col

    def move(self, board):
        if self.set_win(board):
            return self.set_win(board)
        elif self.set_blockOpponentWin(board):
            return self.set_blockOpponentWin(board)
        else:
            return self.set_random(board)


