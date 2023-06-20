- Train q-learn algorithm

in Play.py

```
game_thread = PlayGame()
game_thread.teach(iters)
```

- Run algorithms

in Play.py

```
print("minimax VS baseline 20 times")
game_thread.playMinimaxVSBaseline(20)
print("")

print("minimax VS baseline 50 times")
game_thread.playMinimaxVSBaseline(50)
print("")

print("minimax VS baseline 100 times")
game_thread.playMinimaxVSBaseline(100)
print("")


print("q-learn VS baseline 20 times")
game_thread.playQlearnVSBaseline(20)
print("")

print("q-learn VS baseline 50 times")
game_thread.playQlearnVSBaseline(50)
print("")

print("q-learn VS baseline 100 times")
game_thread.playQlearnVSBaseline(100)
print("")


print("q-learn VS minimax 20 times")
game_thread.playQlearnVSMinimax(20)
print("")

print("q-learn VS minimax 50 times")
game_thread.playQlearnVSMinimax(50)
print("")

print("q-learn VS minimax 100 times")
game_thread.playQlearnVSMinimax(100)
print("")
```

