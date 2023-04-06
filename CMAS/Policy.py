import numpy as np
from scipy.interpolate import interpolate


class policy():
    def __init__(self, state_env, gamma_size, belief_res, epsilon):
        self.gamma_size = gamma_size
        self.belief_res = belief_res
        self.env = state_env
        self.epsilon = epsilon

    def greedy_gamma(self, Q, belief):

        q = np.zeros(self.gamma_size)
        for i in range(self.gamma_size):
            q[i] = interpolate.interp2d(np.linspace(0, 1, self.belief_res), np.linspace(0, 1, self.belief_res), Q[:, :, i])(belief[0], belief[1])

        g_idx = np.argmax(q)
        gamma = self.env.get_gamma(g_idx)

        return q[g_idx], q, g_idx, gamma

    def greedy_gamma1(self, Q, belief):

        b_idx1 = int(belief[0] * self.belief_res) if belief[0] < 1 else self.belief_res - 1
        b_idx2 = int(belief[1] * self.belief_res) if belief[1] < 1 else self.belief_res - 1

        q = Q[b_idx1, b_idx2, :]

        self.g_idx = np.argmax(q)
        gamma = self.env.get_gamma(self.g_idx)

        return q[self.g_idx], q, self.g_idx, gamma

    def e_greedy_gamma(self, g_idx):

        a = np.random.choice([0, 1], p=[self.epsilon, 1 - self.epsilon])
        if a:
            idx = g_idx
        else:
            idx = np.random.randint(0, self.gamma_size)

        gamma = self.env.get_gamma(idx)

        return idx, gamma
