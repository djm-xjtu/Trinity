import os
import pickle
from Agent import QLearner
from Game import train, runQlearnVSMinimax, runQlearnVSBaseline, runMinimaxVSBaseline


class PlayGame:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = {}
        if os.path.isfile('q_agent.pkl'):
            with open('q_agent.pkl', 'rb') as f:
                self.agent = pickle.load(f)
        else:
            self.agent = QLearner(alpha, gamma, epsilon)
        self.games_played = 0

    def teach(self, iters):
        train(self.agent, iters)

    def playQlearnVSMinimax(self, iters):
        print('Qlearn VS Minimax...')
        runQlearnVSMinimax(self.agent, iters)

    def playQlearnVSBaseline(self, iters):
        print('Qlearn VS Baseline...')
        runQlearnVSBaseline(self.agent, iters)

    def playMinimaxVSBaseline(self, iters):
        print('Minimax VS Baseline...')
        runMinimaxVSBaseline(iters)


if __name__ == '__main__':
    game_thread = PlayGame()

    # print("q-learn VS baseline 20 times")
    # game_thread.playQlearnVSBaseline(20)
    # print("")
    #
    # print("q-learn VS baseline 50 times")
    # game_thread.playQlearnVSBaseline(50)
    # print("")
    #
    # print("q-learn VS baseline 100 times")
    # game_thread.playQlearnVSBaseline(100)
    # print("")

    # print("=====================================")
    #
    # print("minimax VS baseline 20 times")
    # game_thread.playMinimaxVSBaseline(20)
    # print("")
    #
    # print("minimax VS baseline 50 times")
    # game_thread.playMinimaxVSBaseline(50)
    # print("")
    #
    # print("minimax VS baseline 100 times")
    # game_thread.playMinimaxVSBaseline(100)
    # print("")
    #
    # print('=====================================')
    #
    print("q-learn VS minimax 20 times")
    game_thread.playQlearnVSMinimax(20)
    print("")

    print("q-learn VS minimax 50 times")
    game_thread.playQlearnVSMinimax(50)
    print("")

    print("q-learn VS minimax 100 times")
    game_thread.playQlearnVSMinimax(100)
    print("")