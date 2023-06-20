import math

from pyamaze import maze, agent, textLabel, COLOR
from queue import PriorityQueue
import time

def h1(cur, goal):
    return math.sqrt((cur[0]-goal[0])**2 + (cur[1]-goal[1])**2)


def h(cur, goal):
    return abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])


def aStar(m):
    st = time.time()
    dir = 'ESWN'
    start = (m.rows, m.cols)
    # The cost path from the start node to the current node
    g_score = {}
    # The cost path from the current node to goal node
    f_score = {}
    for cur in m.grid:
        g_score[cur] = float('inf')
    for cur in m.grid:
        f_score[cur] = float('inf')
    g_score[start] = 0
    f_score[start] = h(start, (1, 1))
    pq = PriorityQueue()
    pq.put((h(start, (1, 1)), h(start, (1, 1)), start))
    path = {}
    while not pq.empty():
        cur = pq.get()[2]
        if cur == (1, 1):
            break
        for d in dir:
            if m.maze_map[cur][d]:
                if d == 'E':
                    Next = (cur[0], cur[1] + 1)
                elif d == 'S':
                    Next = (cur[0] + 1, cur[1])
                elif d == 'W':
                    Next = (cur[0], cur[1] - 1)
                elif d == 'N':
                    Next = (cur[0] - 1, cur[1])
                next_g = g_score[cur] + 1
                next_f = next_g + h(Next, (1, 1))
                if next_f < f_score[Next]:
                    g_score[Next] = next_g
                    f_score[Next] = next_f
                    pq.put((next_f, h(Next, (1, 1)), Next))
                    path[Next] = cur
    astar_path = {}
    cur = (1, 1)
    while cur != start:
        astar_path[path[cur]] = cur
        cur = path[cur]
    ed = time.time()
    return astar_path, ed - st


if __name__ == '__main__':
    m = maze(30, 40)
    m.CreateMaze(loadMaze='maze3.csv',theme=COLOR.light)
    path, spent_time = aStar(m)
    print(path)
    a = agent(m, footprints=True, filled=True)
    m.tracePath({a: path})
    l = textLabel(m, 'A Star Path Length', len(path) + 1)
    l = textLabel(m, 'Spent time', spent_time)
    m.run()
