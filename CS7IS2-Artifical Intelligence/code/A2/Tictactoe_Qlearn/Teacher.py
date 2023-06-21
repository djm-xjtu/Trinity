import random
import Game


class Teacher:
    def set_win(self, board, player=1):
        cells = Game.empty_cells(board)
        for cell in cells:
            if board[cell[0]][cell[1]] == 0:
                board[cell[0]][cell[1]] = player
                if Game.wins(board, player):
                    board[cell[0]][cell[1]] = 0
                    return cell[0], cell[1]
                else:
                    board[cell[0]][cell[1]] = 0

    def set_blockOpponentWin(self, board):
        return self.set_win(board, 2)

    def set_twoThreatToWin(self, board):
        if board[1][0] == 1 and board[0][1] == 1:
            if board[0][0] == 0 and board[2][0] == 0 and board[0][2] == 0:
                return 0, 0
            elif board[1][1] == 0 and board[2][1] == 0 and board[1][2] == 0:
                return 1, 1
        elif board[1][0] == 1 and board[2][1] == 1:
            if board[2][0] == 0 and board[0][0] == 0 and board[2][2] == 0:
                return 2, 0
            elif board[1][1] == 0 and board[0][1] == 0 and board[1][2] == 0:
                return 1, 1
        elif board[2][1] == 1 and board[1][2] == 1:
            if board[2][2] == 0 and board[2][0] == 0 and board[0][2] == 0:
                return 2, 2
            elif board[1][1] == 0 and board[1][0] == 0 and board[0][1] == 0:
                return 1, 1
        elif board[1][2] == 1 and board[0][1] == 1:
            if board[0][2] == 0 and board[0][0] == 0 and board[2][2] == 0:
                return 0, 2
            elif board[1][1] == 0 and board[1][0] == 0 and board[2][1] == 0:
                return 1, 1

        elif board[0][0] == 1 and board[2][2] == 1:
            if board[1][0] == 0 and board[2][1] == 0 and board[2][0] == 0:
                return 2, 0
            elif board[0][1] == 0 and board[1][2] == 0 and board[0][2] == 0:
                return 0, 2
        elif board[2][0] == 1 and board[0][2] == 1:
            if board[2][1] == 0 and board[1][2] == 0 and board[2][2] == 0:
                return 2, 2
            elif board[1][0] == 0 and board[0][1] == 0 and board[0][0] == 0:
                return 0, 0
        return None

    def set_blockOpponentTwoThreatWin(self, board):
        corners = [board[0][0], board[2][0], board[0][2], board[2][2]]
        if board[1][0] == 2 and board[0][1] == 2:
            if board[0][0] == 0 and board[2][0] == 0 and board[0][2] == 0:
                return 0, 0
            elif board[1][1] == 0 and board[2][1] == 0 and board[1][2] == 0:
                return 1, 1
        elif board[1][0] == 2 and board[2][1] == 2:
            if board[2][0] == 0 and board[0][0] == 0 and board[2][2] == 0:
                return 2, 0
            elif board[1][1] == 0 and board[0][1] == 0 and board[1][2] == 0:
                return 1, 1
        elif board[2][1] == 2 and board[1][2] == 2:
            if board[2][2] == 0 and board[2][0] == 0 and board[0][2] == 0:
                return 2, 2
            elif board[1][1] == 0 and board[1][0] == 0 and board[0][1] == 0:
                return 1, 1
        elif board[1][2] == 2 and board[0][1] == 2:
            if board[0][2] == 0 and board[0][0] == 0 and board[2][2] == 0:
                return 0, 2
            elif board[1][1] == 0 and board[1][0] == 0 and board[2][1] == 0:
                return 1, 1
        # if we have two corners, try to set the center
        elif corners.count(0) == 1 and corners.count(2) == 2:
            return 1, 2
        elif board[0][0] == 2 and board[2][2] == 2:
            if board[1][0] == 0 and board[2][1] == 0 and board[2][0] == 0:
                return 2, 0
            elif board[0][1] == 0 and board[2][1] == 0 and board[0][2] == 0:
                return 0, 2
        elif board[2][0] == 2 and board[0][2] == 2:
            if board[2][1] == 0 and board[1][2] == 0 and board[2][2] == 0:
                return 2, 2
            elif board[1][0] == 0 and board[0][1] == 0 and board[0][0] == 0:
                return 0, 0
        return None

    def set_center(self, board):
        if board[1][1] == 0:
            return 1, 1
        return None

    def set_corner(self, board):
        # pick opposite corner
        if board[0][0] == 2 and board[2][2] == 0:
            return 2, 2
        elif board[2][2] == 2 and board[0][0] == 0:
            return 0, 0
        elif board[0][2] == 2 and board[2][0] == 0:
            return 2, 0
        elif board[2][0] == 2 and board[0][2] == 0:
            return 0, 2

        if board[0][0] == 0:
            return 0, 0
        elif board[2][0] == 0:
            return 2, 0
        elif board[0][2] == 0:
            return 0, 2
        elif board[2][2] == 0:
            return 2, 2
        return None

    def set_other(self, board):
        if board[1][0] == 0:
            return 1, 0
        elif board[2][1] == 0:
            return 2, 1
        elif board[1][2] == 0:
            return 1, 2
        elif board[0][1] == 0:
            return 0, 1
        return None

    def set_random(self, board):
        while True:
            x = random.randint(0, 2)
            y = random.randint(0, 2)
            if board[x][y] == 0:
                return x, y

    def move(self, board):
        if random.random() > 0.8:
            return self.set_random(board)
        if self.set_win(board):
            return self.set_win(board)
        if self.set_blockOpponentWin(board):
            return self.set_blockOpponentWin(board)
        if self.set_blockOpponentTwoThreatWin(board):
            return self.set_blockOpponentTwoThreatWin(board)
        if self.set_center(board):
            return self.set_center(board)
        if self.set_corner(board):
            return self.set_corner(board)
        if self.set_other(board):
            return self.set_other(board)
        return self.set_random(board)
