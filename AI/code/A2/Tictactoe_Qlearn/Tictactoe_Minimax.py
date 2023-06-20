import random
from math import inf as infinity
from random import choice

HUMAN = -1
COMP = +1
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]
WIN = 0
LOSE = 0
DRAW = 0


def evaluate(state):
    if wins(state, COMP):
        score = +1
    elif wins(state, HUMAN):
        score = -1
    else:
        score = 0

    return score


def wins(state, player):
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False


def game_over(state):
    return wins(state, HUMAN) or wins(state, COMP)


def empty_cells(state):
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells


def valid_move(x, y):
    if [x, y] in empty_cells(board):
        return True
    else:
        return False


def set_move(x, y, player):
    if valid_move(x, y):
        board[x][y] = player
        return True
    else:
        return False


def minimax(state, depth, player):
    if player == COMP:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] > best[2]:
                best = score
        else:
            if score[2] < best[2]:
                best = score

    return best


def ai_turn(player):
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return
    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
    else:
        move = minimax(board, depth, player)
        x, y = move[0], move[1]
    set_move(x, y, player)


def baseline(player):
    depth = len(empty_cells(board))
    if depth == 0 or game_over(board):
        return

    cells = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    x = -1
    y = -1
    for i in range(len(cells)):
        if 0 <= i <= 2:
            if cells[i][0] == cells[i][1] and cells[i][0] == player and cells[i][2] == 0:
                x = i
                y = 2
                break
            elif cells[i][0] == cells[i][2] and cells[i][0] == player and cells[i][1] == 0:
                x = i
                y = 1
                break
            elif cells[i][1] == cells[i][2] and cells[i][1] == player and cells[i][0] == 0:
                x = i
                y = 0
                break
        elif 3 <= i <= 5:
            if cells[i][0] == cells[i][1] and cells[i][0] == player and cells[i][2] == 0:
                x = 2
                y = i - 3
                break
            elif cells[i][0] == cells[i][2] and cells[i][0] == player and cells[i][1] == 0:
                x = 1
                y = i - 3
                break
            elif cells[i][1] == cells[i][2] and cells[i][1] == player and cells[i][0] == 0:
                x = 0
                y = i - 3
                break
        elif i == 6:
            if cells[i][0] == cells[i][1] and cells[i][0] == player and cells[i][2] == 0:
                x = 2
                y = 2
                break
            elif cells[i][0] == cells[i][2] and cells[i][0] == player and cells[i][1] == 0:
                x = 1
                y = 1
                break
            elif cells[i][1] == cells[i][2] and cells[i][1] == player and cells[i][0] == 0:
                x = 0
                y = 0
                break

    if x == -1 and y == -1:
        while True:
            x = choice([0, 1, 2])
            y = choice([0, 1, 2])
            if valid_move(x, y):
                break
    set_move(x, y, player)


def changeWIN():
    global WIN
    WIN += 1


def changeLOSE():
    global LOSE
    LOSE += 1


def changeDRAW():
    global DRAW
    DRAW += 1


def changeBOARD():
    global board
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]


def reset():
    global WIN, LOSE, DRAW, board
    WIN, LOSE, DRAW = 0, 0, 0


def gameplay():
    while wins(board, HUMAN) == False and wins(board, COMP) == False and len(empty_cells(board)) > 0:
        ai_turn(HUMAN)
        baseline(COMP)
        if wins(board, HUMAN):
            changeWIN()
            break
        elif wins(board, COMP):
            changeLOSE()
            break
        elif len(empty_cells(board)) == 0:
            changeDRAW()
            break


def minimaxVSbaseline(first_player):
    if first_player == HUMAN:
        baseline(HUMAN)
    while True:
        # agent move
        ai_turn(COMP)
        if wins(board, COMP):
            changeWIN()
            break
        elif len(empty_cells(board)) == 0:
            changeDRAW()
            break
        # teacher move
        baseline(HUMAN)
        if wins(board, HUMAN):
            changeLOSE()
            break
        elif len(empty_cells(board)) == 0:
            changeDRAW()
            break


def runMinimaxVSBaseline(iters):
    for i in range(iters):
        if random.random() < 0.5:
            minimaxVSbaseline(HUMAN)
        else:
            minimaxVSbaseline(COMP)
        changeBOARD()
    print("Minimax Win rate: " + str(WIN / iters * 100) + "%")
    print("Minimax Lose rate: " + str(LOSE / iters * 100) + "%")
    print("Minimax Draw rate: " + str(DRAW / iters * 100) + "%")


if __name__ == '__main__':
    print("Minimax VS Baseline 20 times")
    runMinimaxVSBaseline(20)
    print("")
    reset()

    print("Minimax VS Baseline 50 times")
    runMinimaxVSBaseline(50)
    print("")
    reset()

    print("Minimax VS Baseline 100 times")
    runMinimaxVSBaseline(100)
    print("")
    reset()