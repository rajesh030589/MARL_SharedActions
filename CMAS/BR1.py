import numpy as np
from Model import SmartGrid
from tqdm import tqdm
from scipy.interpolate import interpolate
import pickle


def model_seq(RunsI, belief_res, TimeI):

    discount = .9

    gamma_size = 81

    belief1 = np.linspace(0, 1, belief_res)
    belief2 = np.linspace(0, 1, belief_res)

    state_vec_total = [[0, 0], [0, 1], [1, 0], [1, 1]]

    model = SmartGrid(2, belief_res)

    V = np.zeros((TimeI, belief_res, belief_res))
    Q = np.zeros((TimeI, belief_res, belief_res, gamma_size))
    G = np.zeros((belief_res, belief_res, 2))

    for t in tqdm(range(TimeI)):
        for b1 in range(belief_res):
            for b2 in range(belief_res):
                belief = [belief1[b1], belief2[b2]]
                for g_idx in range(gamma_size):
                    belief_prob = [belief[0]*belief[1], belief[0]*(1-belief[1]), (1-belief[0])*belief[1], (1-belief[0])*(1-belief[1])]
                    gamma = model.get_gamma(g_idx)
                    Rew = 0
                    for i in range(4):
                        state_vec = state_vec_total[i]
                        R, A, _ = model.excite_model(state_vec, gamma)
                        next_belief = model.Nxt_belief_vec(belief, gamma, A)
                        if t == 0:
                            Vt = 0
                        else:
                            Vt = interpolate.interp2d(belief1, belief2, V[t - 1, :, :])(next_belief[0], next_belief[1])
                        Rew = Rew + (R + discount*Vt)*belief_prob[i]
                    Q[t, b1, b2, g_idx] = Rew

                gamma_opt_idx = np.argmax(Q[t, b1, b2, :])
                gamma_opt = model.get_gamma(gamma_opt_idx)
                G[b1, b2, 0] = np.random.choice(np.arange(3), p=gamma_opt[0][:, 0])
                G[b1, b2, 1] = np.random.choice(np.arange(3), p=gamma_opt[0][:, 1])
                V[t, b1, b2] = Q[t, b1, b2, gamma_opt_idx]

    with open('/home/raj/Dropbox/MARL/Python Codes/Final_Codes/Data/model_seq.pkl', 'wb') as f:
        pickle.dump([Q, V, G], f)
    f.close()


def model_rnd(RunsI, belief_res, TimeI):

    discount = .9

    gamma_size = 81

    belief1 = np.linspace(0, 1, belief_res)
    belief2 = np.linspace(0, 1, belief_res)

    model = SmartGrid(2, belief_res)

    V_i = np.zeros((RunsI, TimeI, belief_res, belief_res))
    Q_i = np.zeros((RunsI, TimeI, belief_res, belief_res, gamma_size))

    for run in tqdm(range(RunsI)):
        for t in tqdm(range(1, TimeI, 1)):
            for b1 in range(belief_res):
                for b2 in range(belief_res):
                    belief = [belief1[b1], belief2[b2]]
                    for g_idx in range(gamma_size):

                        gamma = model.get_gamma(g_idx)
                        Rew = 0
                        for i in range(10):
                            R, A = model.excite_model(belief, gamma)
                            next_belief = model.Nxt_belief_vec(belief, gamma, A)
                            Vt = interpolate.interp2d(belief1, belief2, V_i[run, t - 1, :, :])(next_belief[0], next_belief[1])
                            Rew = (i*Rew + R + discount*Vt)/(i+1)
                        Q_i[run, t, b1, b2, g_idx] = Rew

                    V_i[run, t, b1, b2] = Q_i[run, t, b1, b2, np.random.randint(0, 81)]
    V = np.mean(V_i, axis=0)
    Q = np.mean(Q_i, axis=0)

    with open('/home/raj/Dropbox/MARL/Python Codes/Final_Codes/Data/model_rnd.pkl', 'wb') as f:
        pickle.dump([Q, V], f)
    f.close()
