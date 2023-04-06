import numpy as np
import scipy.stats as stats
from Model import SmartGrid


class ParticleFilter():
    def __init__(self, num_agents, belief_res, sample_size):
        self.model = SmartGrid(num_agents, belief_res)
        self.num_agents = num_agents
        self.sample_size = sample_size

    def __call__(self, belief_vec, gamma_vec, action_vec):

        belief_vec_n = []
        for i in range(self.num_agents):

            gamma = gamma_vec[i]
            action = action_vec[i]
            belief = belief_vec[i]

            belief_n = self.boostrap_filter(gamma, belief, action)

            belief_vec_n.append(belief_n)

        return belief_vec_n

    def boostrap_filter(self, gamma, belief, action):

        # 1 Sampling step
        M = self.sample_size
        X1 = np.zeros(M, dtype=int)
        # W = np.zeros(M)

        if (1 - belief) < 0:
            belief = 1
        if belief < 0:
            belief = 0
        X1 = stats.bernoulli.rvs(1 - belief, size=M)
        a0 = np.nonzero(gamma[:, 0])
        a1 = np.nonzero(gamma[:, 1])

        A1 = np.where(X1 == 0, a0, a1)[0]
        X2 = self.model.next_state(X1, A1)

        Q = np.zeros(M)
        Q[(X2 == 0) & (A1 == action)] = 1
        L = len(np.where(A1 == action)[0])
        m = sum(Q)
        if L == 0:
            belief_n = 0
        else:
            belief_n = m/L
        return belief_n

    # def particle_filter(self, gamma, belief, action):

    #     # 1 Sampling step
    #     M = 3000
    #     X1 = np.zeros(M, dtype=int)
    #     X2 = np.zeros(M, dtype=int)
    #     A1 = np.zeros(M, dtype=int)
    #     W = np.zeros(M)
    #     for i in range(M):
    #         X1[i] = np.random.choice(np.arange(self.num_states), p=[belief, 1-belief])
    #         A1[i] = np.random.choice(np.arange(self.num_actions), p=gamma[:, X1[i]])
    #         p1 = self.trans_prob(A1[i], 0, X1[i])
    #         X2[i] = np.random.choice(np.arange(self.num_states), p=[p1, 1-p1])

    #     for i in range(M):
    #         m = 0
    #         for j in range(M):
    #             if X2[j] == X2[i] and A1[j] == action:
    #                 m = m + 1
    #         W[i] = m/M

    #     W = W/sum(W)

    #     X = np.zeros(M)
    #     for i in range(M):
    #         a = int(np.random.choice(np.arange(M), p=W))
    #         X[i] = X2[a]

    #     belief_n = 1 - sum(X)/M
    #     return belief_n
