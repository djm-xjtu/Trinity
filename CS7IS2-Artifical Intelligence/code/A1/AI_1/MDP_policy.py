from pyamaze import maze, agent, textLabel, COLOR
import numpy as np
import time


def mdp_policy(m):
    st = time.time()
    dir = 'ESWN'
    n_actions = 4
    size = max(m.cols, m.rows)
    n_states = size**2
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    gamma = 0.8
    theta = 0.001
    probabilities = np.zeros((n_states, n_actions, n_states))
    rewards = np.zeros((n_states, n_actions, n_states))
    for i in range(m.rows):
        for j in range(m.cols):
            for u, d in enumerate(dir):
                if m.maze_map[(i + 1, j + 1)][d]:
                    next_i = i + actions[u][0]
                    next_j = j + actions[u][1]
                    rewards[i*size+j, u, next_i*size+next_j] = 1 if (next_i, next_j) == (m.rows - 1, m.cols - 1) else 0
                else:
                    if m.rows > i + actions[u][0] >= 0 and m.cols > j + actions[u][1] >= 0:
                        rewards[i*size+j, u, next_i*size+next_j] = -10
    for s in range(n_states):
        probabilities[s, 0, s - size if s - size >= 0 else s] = 1.0  # Up
        probabilities[s, 1, s + size if s + size < n_states else s] = 1.0  # Down
        probabilities[s, 2, s - 1 if s % size != 0 else s] = 1.0  # Left
        probabilities[s, 3, s + 1 if (s + 1) % size != 0 else s] = 1.0  # Right
    policy = np.zeros(n_states).astype('int')
    Value = np.zeros(n_states).astype('float')
    is_convergence = False

    while not is_convergence:
        while True:
            delta = 0
            for s in range(n_states):
                v = Value[s]
                Value[s] = np.sum(probabilities[s, policy[s]] * (rewards + gamma * Value))
                print(Value[s])
                delta = max(delta, np.abs(v - Value[s]))
            if delta < theta:
                break
        policy_convergence = True
        for s in range(n_states):
            old_action = policy[s]
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = np.sum(probabilities[s, a] * (rewards + gamma * Value))
            policy[s] = np.argmax(q_values)
            if old_action != policy[s]:
                policy_convergence = False
        if policy_convergence:
            is_convergence = True

    for i in range(m.rows):
        for j in range(m.cols):
            if policy[i][j] == 0:
                print(dir[3], end=' ')
            elif policy[i][j] == 1:
                print(dir[1], end=' ')
            elif policy[i][j] == 2:
                print(dir[2], end=' ')
            elif policy[i][j] == 3:
                print(dir[0], end=' ')
        print("")

    path = [(1, 1)]
    i, j = 0, 0
    while (i < m.rows - 1) and (j < m.cols - 1):
        idx = int(policy[i, j])
        i += actions[idx][0]
        j += actions[idx][1]
        path.append((i + 1, j + 1))

    mdpPolicy_path = {}
    for i in range(0, len(path) - 1):
        mdpPolicy_path[path[i + 1]] = path[i]
    ed = time.time()
    return mdpPolicy_path, ed - st


if __name__ == '__main__':
    m = maze(10, 10)
    m.CreateMaze(loadMaze='maze1.csv', theme=COLOR.light)
    path, spent_time = mdp_policy(m)
    a = agent(m, footprints=True, filled=True)
    m.tracePath({a: path})
    l = textLabel(m, 'Length of Shortest Path', len(path) + 1)
    l = textLabel(m, 'Spent time', spent_time)
    m.run()