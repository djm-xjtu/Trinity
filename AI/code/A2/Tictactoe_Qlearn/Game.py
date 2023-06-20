import math
import random
from math import inf as infinity
from random import choice
import numpy as np
from Teacher import Teacher
HUMAN = 1
COMP = 2
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]
WIN = 0
LOSE = 0
DRAW = 0
GAME_COUNT = 0

WIN_SCORE = 1000000000
LOSE_SCORE = -100000000000000000000000


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
        score = evaluate(state, player)
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


def is_terminal(state):
    return wins(state, COMP) or wins(state, HUMAN) or len(empty_cells(state)) == 0


def minimax1(state, depth, alpha, beta, isCOMP):
    possibleMoves = empty_cells(state)
    if is_terminal(state) or depth == 0:
        if is_terminal(state):
            if wins(state, COMP):
                score = WIN_SCORE + depth * 3
                return (None, score)
            elif wins(state, HUMAN):
                score = LOSE_SCORE - depth * 3
                return (None, score)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, COMP))

    if isCOMP:
        value = -math.inf
        row, col = random.choice(possibleMoves)

        for x, y in possibleMoves:
            new_board = np.copy(state)
            set_move(x, y, COMP)
            newScore = minimax1(new_board, depth - 1, alpha, beta, False)[1]
            if newScore > value:
                value = newScore
                row = x
                col = y
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return row, col, value

    else:
        value = math.inf
        row, col = random.choice(possibleMoves)
        for x, y in possibleMoves:
            new_board = np.copy(state)
            set_move(new_board, col, HUMAN)
            newScore = minimax1(new_board, depth - 1, alpha, beta, True)[1]
            if newScore < value:
                value = newScore
                row = x
                col = y
            beta = min(beta, value)
            if alpha >= beta:
                break

        return row, col, value


def score_position(state, player):
    score = 0

    for r in range(3):
        row_array = [state[r][i] for i in range(3)]
        for c in range(3):
            window = row_array[c:c + 3]
            score += evaluate(window, player)

    for c in range(3):
        col_array = [state[i][c] for i in range(6)]
        for r in range(3):
            window = col_array[r:r + 3]
            score += evaluate(window, player)

    for r in range(3):
        for c in range(3):
            window = [state[r + i][c + i] for i in range(3)]
            score += evaluate(window, player)

    for r in range(3):
        for c in range(3):
            window = [state[r + 3 - i][c + i] for i in range(3)]
            score += evaluate(window, player)

    return score


def evaluate(state, player):
    score = 0
    opponent_count = HUMAN
    if player == HUMAN:
        opponent_count = COMP

    if state.count(player) == 4:
        score += 10000
        return score
    elif state.count(player) == 3 and state.count(0) == 1:
        score += 10
    elif state.count(player) == 2 and state.count(0) == 2:
        score += 3

    if state.count(opponent_count) == 4:
        score -= 10000
        return score
    elif state.count(opponent_count) == 3 and state.count(0) == 1:
        score -= 10
    elif state.count(opponent_count) == 2 and state.count(0) == 2:
        score -= 3
    return score



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


def changeGAME_COUNT():
    global GAME_COUNT
    GAME_COUNT += 1


def changeBoard():
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


def trainGamePlay(first_player, teacher, agent):
    if first_player == HUMAN:
        action = teacher.move(board)
        set_move(action[0], action[1], HUMAN)

    prev_board = toString(board)
    prev_action = agent.get_action(prev_board)

    while True:
        # agent move
        set_move(prev_action[0], prev_action[1], COMP)
        if wins(board, COMP):
            reward = 1
            break
        elif len(empty_cells(board)) == 0:
            reward = 0
            break
        # teacher move
        action = teacher.move(board)
        set_move(action[0], action[1], HUMAN)
        if wins(board, HUMAN):
            reward = -1
            break
        elif len(empty_cells(board)) == 0:
            reward = 0
            break
        else:
            reward = 0
        new_board = toString(board)
        new_action = agent.get_action(new_board)
        agent.update(prev_board, new_board, prev_action, reward)
        prev_board = new_board
        prev_action = new_action

    agent.update(prev_board, None, prev_action, reward)


def teacherPlay(agent):
    teacher = Teacher()
    if random.random() < 0.5:
        trainGamePlay(HUMAN, teacher, agent)
    else:
        trainGamePlay(COMP, teacher, agent)


def train(agent, iters):
    while GAME_COUNT < iters:
        teacherPlay(agent)
        changeGAME_COUNT()
        if GAME_COUNT % 1000 == 0:
            print("Games played: %i" % GAME_COUNT)
        changeBoard()
    agent.save('q_agent.pkl')


def toString(board):
    ans = ''
    for row in board:
        for col in row:
            ans += str(col)
    return ans

# qlearn is COMP
def qlearnVSminimax(first_player, agent):
    if first_player == HUMAN:
        ai_turn(HUMAN)
    prev_board = toString(board)
    prev_action = agent.get_action(prev_board)

    while True:
        # agent move
        set_move(prev_action[0], prev_action[1], COMP)
        if wins(board, COMP):
            changeWIN()
            reward = 1
            break
        elif len(empty_cells(board)) == 0:
            changeDRAW()
            reward = 0
            break
        # teacher move
        ai_turn(HUMAN)
        if wins(board, HUMAN):
            reward = -1
            changeLOSE()
            break
        elif len(empty_cells(board)) == 0:
            reward = 0
            changeDRAW()
            break
        else:
            reward = 0
        new_board = toString(board)
        new_action = agent.get_action(new_board)
        agent.update(prev_board, new_board, prev_action, reward)
        prev_board = new_board
        prev_action = new_action

    agent.update(prev_board, None, prev_action, reward)


def runQlearnVSMinimax(agent, iters):
    while GAME_COUNT < iters:
        if random.random() < 0.5:
            qlearnVSminimax(HUMAN, agent)
        else:
            qlearnVSminimax(COMP, agent)
        changeGAME_COUNT()
        changeBoard()
    print("Qlearn Win rate: " + str(WIN / iters * 100) + "%")
    print("Qlearn Lose rate: " + str(LOSE / iters * 100) + "%")
    print("Qlearn Draw rate: " + str(DRAW / iters * 100) + "%")



def qlearnVSbaseline(first_player, agent):
    if first_player == HUMAN:
        baseline(HUMAN)
    prev_board = toString(board)
    prev_action = agent.get_action(prev_board)

    while True:
        # agent move
        set_move(prev_action[0], prev_action[1], COMP)
        if wins(board, COMP):
            changeWIN()
            reward = 1
            break
        elif len(empty_cells(board)) == 0:
            changeDRAW()
            reward = 0
            break
        # teacher move
        baseline(HUMAN)
        if wins(board, HUMAN):
            reward = -1
            changeLOSE()
            break
        elif len(empty_cells(board)) == 0:
            reward = 0
            changeDRAW()
            break
        else:
            reward = 0
        new_board = toString(board)
        new_action = agent.get_action(new_board)
        agent.update(prev_board, new_board, prev_action, reward)
        prev_board = new_board
        prev_action = new_action

    agent.update(prev_board, None, prev_action, reward)


def runQlearnVSBaseline(agent, iters):
    while GAME_COUNT < iters:
        if random.random() < 0.5:
            qlearnVSbaseline(HUMAN, agent)
        else:
            qlearnVSbaseline(COMP, agent)
        changeGAME_COUNT()
        changeBoard()
    print("Qlearn Win rate: " + str(WIN / iters * 100) + "%")
    print("Qlearn Lose rate: " + str(LOSE / iters * 100) + "%")
    print("Qlearn Draw rate: " + str(DRAW / iters * 100) + "%")
