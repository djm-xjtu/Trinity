import numpy as np
import time
from pyamaze import maze, agent, textLabel, COLOR


def MDP_value(m):
    st = time.time()
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    dir = 'ESWN'
    V = np.zeros((m.rows, m.cols))
    print(m.rows, m.cols)
    probabilities = np.zeros((m.rows, m.cols, 4, m.rows, m.cols))
    rewards = np.zeros((m.rows, m.cols, 4, m.rows, m.cols))
    gamma = 0.9
    theta = 0.001
    for i in range(m.rows):
        for j in range(m.cols):
            for u, d in enumerate(dir):
                if m.maze_map[(i+1, j+1)][d]:
                    next_i = i + actions[u][0]
                    next_j = j + actions[u][1]
                    probabilities[i, j, u, next_i, next_j] = 1
                    rewards[i, j, u, next_i, next_j] = 1 if (next_i, next_j) == (m.rows-1, m.cols-1) else 0
                else:
                    probabilities[i, j, u, i, j] = 1
    while True:
        delta = 0
        for i in range(m.rows):
            for j in range(m.cols):
                v = V[i, j]
                qs = np.zeros(4)
                for u, action in enumerate(actions):
                    qs[u] = np.sum(probabilities[i, j, u] * (rewards[i, j, u] + gamma * V))
                V[i, j] = np.max(qs)
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break

    policy = np.zeros((m.rows, m.cols))
    for i in range(m.rows):
        for j in range(m.cols):
            q_vals = []
            for idx, action in enumerate(actions):
                q = np.sum(probabilities[i, j, idx, :, :] * (rewards[i, j, idx, :, :] + gamma * V))
                q_vals.append(q)
            policy[i, j] = np.argmax(q_vals)

    path = [(1, 1)]
    i, j = 0, 0
    while (i, j) != (m.rows-1, m.cols-1):
        idx = int(policy[i, j])
        i += actions[idx][0]
        j += actions[idx][1]
        path.append((i+1, j+1))

    mdpValue_path = {}
    for i in range(0, len(path) - 1):
        mdpValue_path[path[i + 1]] = path[i]
    ed = time.time()
    return mdpValue_path, ed - st


if __name__ == '__main__':
    m = maze(20, 20)
    m.CreateMaze(loadMaze='maze3.csv', theme=COLOR.light)
    path, spent_time = MDP_value(m)
    a = agent(m, footprints=True, filled=True)
    m.tracePath({a: path})
    l = textLabel(m, 'Length of Shortest Path', len(path) + 1)
    l = textLabel(m, 'Spent time', spent_time)
    m.run()
