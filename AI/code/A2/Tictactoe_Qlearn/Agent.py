import os
import pickle
import collections
import numpy as np
import random


class QLearner:
    def __init__(self, alpha, gamma, eps, eps_decay=0.):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.actions = []
        for i in range(3):
            for j in range(3):
                self.actions.append((i, j))
        self.Q = {}
        for action in self.actions:
            self.Q[action] = collections.defaultdict(int)
        self.rewards = []

    def get_action(self, s):
        possible_actions = [a for a in self.actions if s[a[0]*3 + a[1]] == '0']
        if random.random() < self.eps:
            action = possible_actions[random.randint(0, len(possible_actions)-1)]
        else:
            values = np.array([self.Q[a][s] for a in possible_actions])
            col_max = np.where(values == np.max(values))[0]
            if len(col_max) > 1:
                col = np.random.choice(col_max, 1)[0]
            else:
                col = col_max[0]
            action = possible_actions[col]

        self.eps *= (1.-self.eps_decay)
        return action

    def save(self, path):
        if os.path.isfile(path):
            os.remove(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def update(self, state, next_state, action, reward):
        if next_state is not None:
            possible_actions = []
            Qs = []
            for action in self.actions:
                if next_state[action[0] * 3 + action[1]] == '0':
                    possible_actions.append(action)
            for action in possible_actions:
                Qs.append(self.Q[action][next_state])
            self.Q[action][state] += self.alpha*(reward + self.gamma*np.max(Qs) - self.Q[action][state])
        else:
            self.Q[action][state] += self.alpha*(reward - self.Q[action][state])
        self.rewards.append(reward)
