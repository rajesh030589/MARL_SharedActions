import numpy as np
import scipy.stats as stats


class SmartGrid():
    def __init__(self, num_agents, belief_res):

        self.M = np.array([[.25, .75],
                           [.375, .625]])

        self.e1 = .2
        self.e2 = .2

        self.c1 = .1
        self.c2 = .2

        # Ideal state distribution
        self.zeta = np.array([.7, .3])

        self.num_states = 2
        self.num_actions = 3

        self.num_agents = num_agents
        self.belief_res = belief_res

        # gamma size
        self.gamma_size = (self.num_actions * self.num_states)**self.num_agents

    def trans_prob(self, action, next_state, state):

        if action == 0:
            return self.M[state, next_state]
        elif action == 1:
            N = self.e1*self.M + (1 - self.e1) * np.array([[1, 0], [1, 0]])
            return N[state, next_state]
        else:
            N = self.e2*self.M + (1 - self.e2) * np.array([[0, 1], [0, 1]])
            return N[state, next_state]

    def belief_state(self, gamma, belief, action):

        F_num = belief * gamma[action, 0] * self.trans_prob(action, 0, 0) + (1-belief) * gamma[action, 1] * self.trans_prob(action, 0, 1)
        F_dem = belief * gamma[action, 0] + (1 - belief) * gamma[action, 1]
        F_num1 = belief * self.trans_prob(action, 0, 0) + (1-belief) * self.trans_prob(action, 0, 1)
        if F_dem != 0:
            F = F_num / F_dem
        else:
            F = F_num1
        return F

    def Nxt_belief_vec(self, belief_vec, gamma_vec, action_vec):

        belief_vec_n = []
        for i in range(self.num_agents):

            gamma = gamma_vec[i]
            action = action_vec[i]
            belief = belief_vec[i]

            belief_n = self.belief_state(gamma, belief, action)

            belief_vec_n.append(belief_n)

        return belief_vec_n

    def KL_diver(self, Z, Zeta):

        if Z[0] == 0:
            K1 = 0
        else:
            K1 = Z[0]*np.log(Z[0]/Zeta[0])

        if Z[1] == 0:
            K2 = 0
        else:
            K2 = Z[1]*np.log(Z[1]/Zeta[1])

        return K1 + K2

    def group_rewards(self, state_vector, action_vector):

        r1 = ((action_vector.count(1))*self.c1)/self.num_agents
        r2 = ((action_vector.count(2))*self.c2)/self.num_agents

        z0 = (state_vector.count(0))/self.num_agents
        z1 = (state_vector.count(1))/self.num_agents

        return -(r1 + r2 + self.KL_diver([z0, z1], self.zeta))

    def get_belief_vec(self):

        belief_vec = []
        b_id_vec = []

        for _ in range(self.num_agents):
            belief = np.random.uniform()
            b_id = int(belief * self.belief_res) if belief < 1 else self.belief_res

            belief_vec.append(belief)
            b_id_vec.append(b_id)

        return b_id_vec, belief_vec

    def random_gamma(self):

        gamma_vec = []
        for _ in range(self.num_agents):
            gamma = np.zeros((self.num_actions, self.num_states))

            for s in range(self.num_states):
                g = int(np.random.choice(np.arange(self.num_actions)))
                gamma[g, s] = 1
            gamma_vec.append(gamma)

        return gamma_vec

    def random_state(self, belief_vec):

        state_vec = []
        for i in range(self.num_agents):
            s = np.random.choice([0, 1], p=[belief_vec[i], 1-belief_vec[i]])
            state_vec.append(s)
        return state_vec

    def get_action_vec(self, state_vec, gamma_vec):

        action_vec = []

        for i in range(self.num_agents):
            s = state_vec[i]
            a = np.random.choice(np.arange(self.num_actions), p=gamma_vec[i][:, s])

            action_vec.append(a)

        return action_vec

    def get_gamma_size(self):
        return self.gamma_size

    def get_gamma(self, idx):
        # N = self.num_agents
        # K = idx
        # a = self.num_actions * self.num_states
        # b = np.zeros(N)

        # for n in range(N - 1, 0, -1):
        #     b[n] = int(K/(a**n))
        #     K = K - int(K/(a**n))

        # b[0] = K

        gamma_list = self.gamma_list()
        # gamma_vec = []
        # for n in range(self.num_agents):
        #     gamma_vec.append(gamma_list[int(b[n])])

        a = int(idx / 9)
        b = int(idx - 9*a)
        gamma_vec = []
        gamma_vec.append(gamma_list[b])
        gamma_vec.append(gamma_list[a])

        return gamma_vec

    def gamma_list(self):

        gamma_list = []

        gamma1 = np.array([[1, 1], [0, 0], [0, 0]])
        gamma2 = np.array([[0, 1], [1, 0], [0, 0]])
        gamma3 = np.array([[0, 1], [0, 0], [1, 0]])
        gamma4 = np.array([[1, 0], [0, 1], [0, 0]])
        gamma5 = np.array([[0, 0], [1, 1], [0, 0]])
        gamma6 = np.array([[0, 0], [0, 1], [1, 0]])
        gamma7 = np.array([[1, 0], [0, 0], [0, 1]])
        gamma8 = np.array([[0, 0], [1, 0], [0, 1]])
        gamma9 = np.array([[0, 0], [0, 0], [1, 1]])

        gamma_list.append(gamma1)
        gamma_list.append(gamma2)
        gamma_list.append(gamma3)
        gamma_list.append(gamma4)
        gamma_list.append(gamma5)
        gamma_list.append(gamma6)
        gamma_list.append(gamma7)
        gamma_list.append(gamma8)
        gamma_list.append(gamma9)

        return gamma_list

    def excite_model(self, state_vec, gamma):

        # state_vec = self.random_state(belief)
        action_vec = self.get_action_vec(state_vec, gamma)

        reward = self.group_rewards(state_vec, action_vec)

        next_state = self.next_state_vec(state_vec, action_vec)

        return reward, action_vec, next_state

    def next_state_vec(self, state_vec, action_vec):

        next_state = []
        for i in range(self.num_agents):
            p = self.trans_prob(action_vec[i], 0, state_vec[i])
            sn = np.random.choice([0, 1], p=[p, 1 - p])
            next_state.append(sn)

        return next_state

    def next_state(self, state, action):

        X1 = state
        A1 = action
        M = len(X1)

        X2 = np.zeros(M, dtype=int)

        P = np.zeros(M)
        P[(X1 == 0) & (A1 == 0)] = .75
        P[(X1 == 1) & (A1 == 0)] = .625
        P[(X1 == 0) & (A1 == 1)] = .150
        P[(X1 == 1) & (A1 == 1)] = .125
        P[(X1 == 0) & (A1 == 2)] = .950
        P[(X1 == 1) & (A1 == 2)] = .925

        L1 = len(np.where(P == .75)[0])
        L2 = len(np.where(P == .625)[0])
        L3 = len(np.where(P == .150)[0])
        L4 = len(np.where(P == .125)[0])
        L5 = len(np.where(P == .950)[0])
        L6 = len(np.where(P == .925)[0])

        G1 = np.zeros(L1)
        G2 = np.zeros(L2)
        G3 = np.zeros(L3)
        G4 = np.zeros(L4)
        G5 = np.zeros(L5)
        G6 = np.zeros(L6)

        L11 = int(L1*0.75)
        L21 = int(L2*0.625)
        L31 = int(L3*0.150)
        L41 = int(L4*0.125)
        L51 = int(L5*0.950)
        L61 = int(L6*0.925)

        G1[0:L11] = np.ones(L11)
        G2[0:L21] = np.ones(L21)
        G3[0:L31] = np.ones(L31)
        G4[0:L41] = np.ones(L41)
        G5[0:L51] = np.ones(L51)
        G6[0:L61] = np.ones(L61)

        X2[(X1 == 0) & (A1 == 0)] = G1
        X2[(X1 == 1) & (A1 == 0)] = G2
        X2[(X1 == 0) & (A1 == 1)] = G3
        X2[(X1 == 1) & (A1 == 1)] = G4
        X2[(X1 == 0) & (A1 == 2)] = G5
        X2[(X1 == 1) & (A1 == 2)] = G6

        return X2
