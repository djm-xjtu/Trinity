import math
import random
import numpy as np
import time
from Teacher import Teacher

board = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
HUMAN = 1
COMP = 2
WIN = 0
LOSE = 0
DRAW = 0

WIN_SCORE = 1e9
LOSE_SCORE = -1e23
GAME_COUNT = 0


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


def getValidColumns(state):
    validColumns = []
    for i in range(7):
        if state[0][i] == 0:
            validColumns.append(i)
    return validColumns


def is_terminal(state):
    return checkWin(state, HUMAN) or checkWin(state, COMP) or len(getValidColumns(state)) == 0


def valid_move(col):
    if board[0][col] == 0:
        return True
    else:
        return False


def minimax(state, depth, alpha, beta, isCOMP):
    possibleMoves = getValidColumns(state)
    if is_terminal(state) or depth == 0:
        if is_terminal(state):
            if checkWin(state, COMP):
                score = WIN_SCORE + depth * 3
                return None, score
            elif checkWin(state, HUMAN):
                score = LOSE_SCORE - depth * 3
                return None, score
            else:
                return None, 0
        else:
            return None, score_position(board, COMP)

    if isCOMP:
        value = -math.inf
        column = random.choice(possibleMoves)
        for col in possibleMoves:
            new_board = np.copy(state)
            set_move(new_board, col, COMP)
            newScore = minimax(new_board, depth - 1, alpha, beta, False)[1]
            if newScore > value:
                value = newScore
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return column, value

    else:
        value = math.inf
        column = random.choice(possibleMoves)
        for col in possibleMoves:
            new_board = np.copy(state)
            set_move(new_board, col, HUMAN)
            newScore = minimax(new_board, depth - 1, alpha, beta, True)[1]
            if newScore < value:
                value = newScore
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break

        return column, value


def score_position(state, player):
    score = 0

    for r in range(6):
        row_array = [state[r][i] for i in range(7)]
        for c in range(4):
            window = row_array[c:c + 4]
            score += evaluate(window, player)

    for c in range(7):
        col_array = [state[i][c] for i in range(6)]
        for r in range(3):
            window = col_array[r:r + 4]
            score += evaluate(window, player)

    for r in range(3):
        for c in range(4):
            window = [state[r + i][c + i] for i in range(4)]
            score += evaluate(window, player)

    for r in range(3):
        for c in range(4):
            window = [state[r + 3 - i][c + i] for i in range(4)]
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
    board = [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]]


def changeGAME_COUNT():
    global GAME_COUNT
    GAME_COUNT += 1


def reset():
    global WIN, LOSE, DRAW, GAME_COUNT
    WIN = 0
    LOSE = 0
    DRAW = 0
    GAME_COUNT = 0


def baseline(player):
    if is_terminal(board):
        return
    column = -1
    valid_columns = getValidColumns(board)
    for valid_column in valid_columns:
        new_board = np.copy(board)
        set_move(new_board, valid_column, player)
        if checkWin(new_board, player):
            column = valid_column
            break
    if column == -1:
        while True:
            column = random.choice([0, 1, 2, 3, 4, 5, 6])
            if valid_move(column):
                break
    set_move(board, column, player)


def testBaseline():
    itr = 0
    while True:
        itr += 1
        baseline(HUMAN)
        if is_terminal(board):
            break
        baseline(COMP)
        if is_terminal(board):
            break


def trainGamePlay(first_player, teacher, agent):
    depth = 6
    round = 0
    if first_player == HUMAN:
        # action = teacher.move(board)
        action = minimax(board, 6, -math.inf, math.inf, True)[0]
        set_move(board, action, HUMAN)

    prev_board = toString(board)
    prev_action = agent.get_action(prev_board)

    while True:
        # agent move
        set_move(board, prev_action, COMP)
        if checkWin(board, COMP):
            reward = 1
            break
        elif len(getValidColumns(board)) == 0:
            reward = 0
            break
        # teacher move
        # action = teacher.move(board)
        # set_move(board, action, HUMAN)
        start_time = time.time()
        ai_col = minimax(board, depth - 1, -math.inf, math.inf, True)[0]
        end_time = time.time()
        set_move(board, ai_col, HUMAN)
        run_time = end_time - start_time
        print("Round " + str(round) + ": Time taken: " + str(run_time) + "s")
        print("")
        if checkWin(board, HUMAN):
            reward = -1
            break
        elif len(getValidColumns(board)) == 0:
            reward = 0
            break
        else:
            reward = 0
        new_board = toString(board)
        new_action = agent.get_action(new_board)
        agent.update(prev_board, new_board, prev_action, reward)
        prev_board = new_board
        prev_action = new_action
        if run_time > 1 and round > 3:
            depth -= 1
        round += 1
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
        changeBOARD()
    agent.save('q_agent_50k.pkl')


# minimax as COMP and baseline as HUMAN
def minimaxVSbaseline(first_player):
    depth = 6
    round = 0
    if first_player == HUMAN:
        baseline(HUMAN)
    while True:
        # agent move
        start_time = time.time()
        ai_col = minimax(board, depth - 1, -math.inf, math.inf, True)[0]
        end_time = time.time()
        set_move(board, ai_col, COMP)
        run_time = end_time - start_time
        print("Round " + str(round) + ": Time taken: " + str(run_time) + "s")
        print("")
        if checkWin(board, COMP):
            changeWIN()
            break
        elif len(getValidColumns(board)) == 0:
            changeDRAW()
            break
        # teacher move
        baseline(HUMAN)
        if checkWin(board, HUMAN):
            changeLOSE()
            break
        elif len(getValidColumns(board)) == 0:
            changeDRAW()
            break

        # if run_time < 7.5 and round > 4:
        #     depth += 1
        # elif run_time > 12.5 and round > 4:
        #     depth -=
        if run_time > 3 and round > 4:
            depth -= 1
        round += 1


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
    reset()


# qlearn as COMP and minimax as HUMAN
def qlearnVSminimax(first_player, agent):
    depth = 6
    round = 0
    if first_player == HUMAN:
        ai_col = minimax(board, depth - 1, -math.inf, math.inf, True)[0]
        set_move(board, ai_col, HUMAN)
    prev_board = toString(board)
    prev_action = agent.get_action(prev_board)
    while True:
        # agent move
        set_move(board, prev_action, COMP)
        if checkWin(board, COMP):
            changeWIN()
            reward = 1
            break
        elif len(getValidColumns(board)) == 0:
            changeDRAW()
            reward = 0
            break
        # teacher move
        start_time = time.time()
        ai_col = minimax(board, depth - 1, -math.inf, math.inf, True)[0]
        end_time = time.time()
        set_move(board, ai_col, HUMAN)
        run_time = end_time - start_time
        print("Round " + str(round) + ": Time taken: " + str(run_time) + "s")
        print("")
        if checkWin(board, HUMAN):
            reward = -1
            changeLOSE()
            break
        elif len(getValidColumns(board)) == 0:
            reward = 0
            changeDRAW()
            break
        else:
            reward = 0

        # if run_time < 7.5 and round > 4:
        #     depth += 1
        # elif run_time > 12.5 and round > 4:
        #     depth -= 1
        if run_time > 3 and round > 4:
            depth -= 1
        round += 1

        new_board = toString(board)
        new_action = agent.get_action(new_board)
        agent.update(prev_board, new_board, prev_action, reward)
        prev_board = new_board
        prev_action = new_action

    agent.update(prev_board, None, prev_action, reward)


def runQlearnVSMinimax(agent, iters):
    for i in range(iters):
        if random.random() < 0.5:
            qlearnVSminimax(HUMAN, agent)
        else:
            qlearnVSminimax(COMP, agent)
        changeBOARD()
    print("Qlearn Win rate: " + str(WIN / iters * 100) + "%")
    print("Qlearn Lose rate: " + str(LOSE / iters * 100) + "%")
    print("Qlearn Draw rate: " + str(DRAW / iters * 100) + "%")
    reset()

def qlearnVSbaseline(first_player, agent):
    if first_player == HUMAN:
        baseline(HUMAN)
    prev_board = toString(board)
    prev_action = agent.get_action(prev_board)

    while True:
        # agent move
        set_move(board, prev_action, COMP)
        if checkWin(board, COMP):
            changeWIN()
            reward = 1
            break
        elif len(getValidColumns(board)) == 0:
            changeDRAW()
            reward = 0
            break
        # teacher move
        baseline(HUMAN)
        if checkWin(board, HUMAN):
            reward = -1
            changeLOSE()
            break
        elif len(getValidColumns(board)) == 0:
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
        changeBOARD()
    print("Qlearn Win rate: " + str(WIN / iters * 100) + "%")
    print("Qlearn Lose rate: " + str(LOSE / iters * 100) + "%")
    print("Qlearn Draw rate: " + str(DRAW / iters * 100) + "%")
    reset()



def toString(board):
    ans = ''
    for row in board:
        for col in row:
            ans += str(col)
    return ans
