from pyamaze import maze, agent, textLabel, COLOR
import time

def DFS(m):
    st = time.time()
    dir = 'ESWN'
    destination = (1, 1)
    start = (m.rows, m.cols)
    vis = [start]
    queue = [start]
    path = {}
    while len(queue) > 0:
        cur = queue.pop()
        if cur == destination:
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
                if Next in vis:
                    continue
                vis.append(Next)
                queue.append(Next)
                path[Next] = cur
    dfs_path = {}
    cur = (1, 1)
    while cur != start:
        dfs_path[path[cur]] = cur
        cur = path[cur]
    ed = time.time()
    print(st, ed)
    return dfs_path, ed - st


if __name__ == '__main__':
    m = maze(20, 20)
    m.CreateMaze(loopPercent=40, theme=COLOR.light, loadMaze='maze3.csv')
    path, spent_time = DFS(m)
    a = agent(m, footprints=True, filled=True)
    m.tracePath({a: path})
    l = textLabel(m, 'Length of Shortest Path', len(path) + 1)
    l = textLabel(m, 'spent_time', spent_time)
    m.run()

