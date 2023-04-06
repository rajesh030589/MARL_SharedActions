import numpy as np
from Model import SmartGrid
from tqdm import tqdm
from Policy import policy
from scipy.interpolate import interpolate
import matplotlib.pyplot as plt
import sys
import pickle
from ParticleFilter import ParticleFilter
from Value_NN import ValueFunctionWithNN


def model_rl(RunsI, EpisodeI, belief_res, TimeI):

    logging = False
    if logging:
        log_file = open('Data/log1.txt', 'wt')
        sys.stdout = log_file

    discount = .9
    # epsilon = .2

    # alpha = .95
    epsilon = .5

    alpha = .95
    gamma_size = 81

    model = SmartGrid(2, belief_res)
    gamma = policy(model, gamma_size, belief_res, epsilon)

    V_i = np.zeros((RunsI, TimeI, belief_res, belief_res))
    Q_i = np.zeros((RunsI, TimeI, belief_res, belief_res, gamma_size))

    Val_i = np.zeros((RunsI, EpisodeI, TimeI))

    # Multiple Runs
    for run in tqdm(range(RunsI)):

        # Each episode
        for e in tqdm(range(EpisodeI)):
            if logging:
                print("Run: ", run, "Episode: ", e)
                print("=========================")

            b_idx, belief = model.get_belief_vec()
            if logging:
                print("Choose initial belief: ", belief[0], belief[1])
                print("index of initial belief: ", b_idx[0], b_idx[1])

            state_vec = model.random_state(belief)

            # Time Iteration starts

            for t in range(TimeI):

                if logging:
                    for idx1 in range(belief_res):
                        for idx2 in range(belief_res):
                            for gdx in range(gamma_size):
                                print("The Q values at ", idx1, idx2, gdx, " is :", Q_i[run, t, idx1, idx2, gdx])

                    for idx1 in range(belief_res):
                        for idx2 in range(belief_res):
                            print("The V value at ", idx1, idx2, " is :", V_i[run, t, idx1, idx2])

                max_q, q, g_idx, greedy_gamma = gamma.greedy_gamma1(Q_i[run, t, :, :, :], belief)

                V_i[run, t, b_idx[0], b_idx[1]] = max_q

                Val_i[run, e, t] = max_q

                if logging:
                    print("Step 1: Policy Choice: Policy Module")
                    print("=====================================")
                    print("Inputs: belief", belief[0], belief[1])
                    print("index of belief: ", b_idx[0], b_idx[1])
                    print("Q_value at the belief", q)
                    print("Best Q: ", max_q)
                    print("Best gamma: ", greedy_gamma, g_idx)

                if logging:
                    print("New Value function at t = ", t, " is ", V_i[run, t, :, :])

                e_idx, e_greedy_gamma = gamma.e_greedy_gamma(g_idx)

                if logging:
                    print("epsilon gamma: ", e_greedy_gamma, e_idx)

                R, A, Xn = model.excite_model(state_vec, e_greedy_gamma)
                next_belief = model.Nxt_belief_vec(belief, e_greedy_gamma, A)

                if logging:
                    print("Step 2: Next Belief Calculation: Particle Filter Module")
                    print("=====================================")
                    print("Next belief at gamma ", e_greedy_gamma, " , ", e_idx, " = ", next_belief[0], next_belief[1])

                if logging:
                    print("Step 3: Expected Reward: Model")
                    print("=====================================")
                    print("Expected Reward at gamma ", e_greedy_gamma, " and belief ", belief[0], belief[1], " = ", R)

                b_1 = int(next_belief[0] * belief_res) if next_belief[0] < 1 else belief_res - 1
                b_2 = int(next_belief[1] * belief_res) if next_belief[1] < 1 else belief_res - 1

                if t == TimeI - 1:
                    Vt = 0
                else:
                    Vt = V_i[run, t+1, b_1, b_2]

                if logging:
                    print("Future state value at e_gamma: ", e_greedy_gamma, " and belief: ", next_belief[0], next_belief[1], " is ", Vt)

                Rew = R + discount*Vt

                if logging:
                    print("Step 4: Compute the new Q")
                    print("==========================")
                    print("The value of OLD Q at ", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx])
                    print("The value added", t, b_idx[0], b_idx[1], e_idx, " is ", Rew)
                    print("The TD Error", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx] - Rew)

                Q_i[run, t, b_idx[0], b_idx[1], e_idx] = Q_i[run, t, b_idx[0], b_idx[1], e_idx] + alpha * (Rew - Q_i[run, t, b_idx[0], b_idx[1], e_idx])
                if logging:
                    print("The value of updated Q at ", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx])

                logging = False
                if logging:
                    print("The non zero values of V: ", np.nonzero(V_i))
                    print("The non zero values of Q: ", np.nonzero(Q_i))

                belief = next_belief
                b_idx[0] = int(belief[0] * belief_res) if belief[0] < 1 else belief_res - 1
                b_idx[1] = int(belief[1] * belief_res) if belief[1] < 1 else belief_res - 1

                state_vec = Xn

    Val = np.mean(Val_i, axis=0)
    Qp = np.mean(Q_i, axis=0)[0, :, :, :]

    with open('Data/model_rl.pkl', 'wb') as f:
        pickle.dump([Val, Qp], f)

    f.close()
    if logging:
        log_file.close()


def modelfree_rl(RunsI, EpisodeI, belief_res, TimeI, filter_size):

    logging = False
    # log_file = open('/home/raj/Dropbox/MARL/Python Codes/Final_Codes/Data/log1.txt', 'wt')
    # sys.stdout = log_file

    num_agents = 2

    discount = .9
    epsilon = .2

    alpha = .95
    gamma_size = 81

    model = SmartGrid(num_agents, belief_res)
    Filter = ParticleFilter(num_agents, belief_res, filter_size)
    gamma = policy(model, gamma_size, belief_res, epsilon)

    V_i = np.zeros((RunsI, TimeI + 1, belief_res, belief_res))
    Q_i = np.zeros((RunsI, TimeI, belief_res, belief_res, gamma_size))
    # G = np.zeros((belief_res, belief_res, 2, 2))
    Val_i = np.zeros((RunsI, EpisodeI, TimeI))

    # Multiple Runs
    for run in range(RunsI):

        # Each episode
        for e in tqdm(range(EpisodeI)):
            if logging:
                print("Run: ", run, "Episode: ", e)
                print("=========================")

            b_idx, belief = model.get_belief_vec()
            if logging:
                print("Choose initial belief: ", belief[0], belief[1])
                print("index of initial belief: ", b_idx[0], b_idx[1])

            state_vec = model.random_state(belief)

            # Time Iteration starts
            # G = []
            for t in range(TimeI):

                if logging:
                    for idx1 in range(belief_res):
                        for idx2 in range(belief_res):
                            for gdx in range(gamma_size):
                                print("The Q values at ", idx1, idx2, gdx, " is :", Q_i[run, t, idx1, idx2, gdx])

                    for idx1 in range(belief_res):
                        for idx2 in range(belief_res):
                            print("The V value at ", idx1, idx2, " is :", V_i[run, t, idx1, idx2])

                max_q, q, g_idx, greedy_gamma = gamma.greedy_gamma1(Q_i[run, t, :, :, :], belief)

                V_i[run, t, b_idx[0], b_idx[1]] = max_q

                Val_i[run, e, t] = max_q

                # G[b_idx[0], b_idx[1], 0, 0] = np.random.choice(np.arange(3), p=greedy_gamma[0][:, 0])
                # G[b_idx[0], b_idx[1], 0, 1] = np.random.choice(np.arange(3), p=greedy_gamma[0][:, 1])

                # G[b_idx[0], b_idx[1], 1, 0] = np.random.choice(np.arange(3), p=greedy_gamma[1][:, 0])
                # G[b_idx[0], b_idx[1], 1, 1] = np.random.choice(np.arange(3), p=greedy_gamma[1][:, 1])

                if logging:
                    print("Step 1: Policy Choice: Policy Module")
                    print("=====================================")
                    print("Inputs: belief", belief[0], belief[1])
                    print("index of belief: ", b_idx[0], b_idx[1])
                    print("Q_value at the belief", q)
                    print("Best Q: ", max_q)
                    print("Best gamma: ", greedy_gamma, g_idx)

                if logging:
                    print("New Value function at t = ", t, " is ", V_i[run, t, :, :])

                e_idx, e_greedy_gamma = gamma.e_greedy_gamma(g_idx)

                if logging:
                    print("epsilon gamma: ", e_greedy_gamma, e_idx)

                R, A, Xn = model.excite_model(state_vec, e_greedy_gamma)
                next_belief = Filter(belief, e_greedy_gamma, A)

                if logging:
                    print("Step 2: Next Belief Calculation: Particle Filter Module")
                    print("=====================================")
                    print("Next belief at gamma ", e_greedy_gamma, " , ", e_idx, " = ", next_belief[0], next_belief[1])

                if logging:
                    print("Step 3: Expected Reward: Model")
                    print("=====================================")
                    print("Expected Reward at gamma ", e_greedy_gamma, " and belief ", belief[0], belief[1], " = ", R)

                if logging:
                    print("Future state value at e_gamma: ", e_greedy_gamma, " and belief: ", next_belief[0], next_belief[1], " is ", Vt)

                b_1 = int(next_belief[0] * belief_res) if next_belief[0] < 1 else belief_res - 1
                b_2 = int(next_belief[1] * belief_res) if next_belief[1] < 1 else belief_res - 1

                if t == TimeI - 1:
                    Vt = 0
                else:
                    Vt = V_i[run, t+1, b_1, b_2]

                Rew = R + discount*Vt

                if logging:
                    print("Step 4: Compute the new Q")
                    print("==========================")
                    print("The value of OLD Q at ", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx])
                    print("The value added", t, b_idx[0], b_idx[1], e_idx, " is ", Rew)
                    print("The TD Error", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx] - Rew)

                Q_i[run, t, b_idx[0], b_idx[1], e_idx] = Q_i[run, t, b_idx[0], b_idx[1], e_idx] + alpha * (Rew - Q_i[run, t, b_idx[0], b_idx[1], e_idx])
                if logging:
                    print("The value of updated Q at ", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx])

                logging = False
                if logging:
                    print("The non zero values of V: ", np.nonzero(V_i))
                    print("The non zero values of Q: ", np.nonzero(Q_i))

                belief = next_belief

                b_idx[0] = int(belief[0] * belief_res) if belief[0] < 1 else belief_res - 1
                b_idx[1] = int(belief[1] * belief_res) if belief[1] < 1 else belief_res - 1

                state_vec = Xn

    Val = np.mean(Val_i, axis=0)

    with open('Data/modelfree_'+str(filter_size)+'_rl.pkl', 'wb') as f:
        # pickle.dump(G, f)
        pickle.dump(Val, f)
    f.close()
    # log_file.close()


def model_rl_nn(RunsI, belief_res, TimeI):

    logging = False
    # log_file = open('/home/raj/Dropbox/MARL/Python Codes/Final_Codes/Data/log1.txt', 'wt')
    # sys.stdout = log_file

    discount = .9
    epsilon = .1

    EpisodeI = 12000

    alpha = .5
    gamma_size = 81

    model = SmartGrid(2, belief_res)
    gamma = policy(model, gamma_size, belief_res, epsilon)

    Value = []
    for _ in range(TimeI):
        Value.append(ValueFunctionWithNN(2))

    # V_i = np.zeros((RunsI, TimeI + 1, belief_res, belief_res))
    Q_i = np.zeros((RunsI, TimeI, belief_res, belief_res, gamma_size))

    Ret = np.zeros((EpisodeI, TimeI))

    # Multiple Runs
    for run in range(RunsI):

        # Each episode
        for e in tqdm(range(EpisodeI)):
            if logging:
                print("Run: ", run, "Episode: ", e)
                print("=========================")

            b_idx, belief = model.get_belief_vec()
            if logging:
                print("Choose initial belief: ", belief[0], belief[1])
                print("index of initial belief: ", b_idx[0], b_idx[1])

            state_vec = model.random_state(belief)

            # Time Iteration starts
            G = []
            for t in range(TimeI):

                if logging:
                    for idx1 in range(belief_res):
                        for idx2 in range(belief_res):
                            for gdx in range(gamma_size):
                                print("The Q values at ", idx1, idx2, gdx, " is :", Q_i[run, t, idx1, idx2, gdx])

                max_q, q, g_idx, greedy_gamma = gamma.greedy_gamma1(Q_i[run, t, :, :, :], belief)

                Value[t].update(max_q, np.array(belief))

                G.append(greedy_gamma)

                if logging:
                    print("Step 1: Policy Choice: Policy Module")
                    print("=====================================")
                    print("Inputs: belief", belief[0], belief[1])
                    print("index of belief: ", b_idx[0], b_idx[1])
                    print("Q_value at the belief", q)
                    print("Best Q: ", max_q)
                    print("Best gamma: ", greedy_gamma, g_idx)

                e_idx, e_greedy_gamma = gamma.e_greedy_gamma(Q_i[run, t, :, :, :], belief)

                if logging:
                    print("epsilon gamma: ", e_greedy_gamma, e_idx)

                R, A, Xn = model.excite_model(state_vec, e_greedy_gamma)
                next_belief = model.Nxt_belief_vec(belief, e_greedy_gamma, A)

                if logging:
                    print("Step 2: Next Belief Calculation: Particle Filter Module")
                    print("=====================================")
                    print("Next belief at gamma ", e_greedy_gamma, " , ", e_idx, " = ", next_belief[0], next_belief[1])

                if logging:
                    print("Step 3: Expected Reward: Model")
                    print("=====================================")
                    print("Expected Reward at gamma ", e_greedy_gamma, " and belief ", belief[0], belief[1], " = ", R)

                if t == 49:
                    Rew = R
                else:
                    Rew = R + discount*Value[t+1](np.array(next_belief))

                Ret[e, t] = Rew

                if logging:
                    print("Step 4: Compute the new Q")
                    print("==========================")
                    print("The value of OLD Q at ", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx])
                    print("The value added", t, b_idx[0], b_idx[1], e_idx, " is ", Rew)
                    print("The TD Error", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx] - Rew)

                Q_i[run, t, b_idx[0], b_idx[1], e_idx] = Q_i[run, t, b_idx[0], b_idx[1], e_idx] + alpha * (Rew - Q_i[run, t, b_idx[0], b_idx[1], e_idx])

                if logging:
                    print("The value of updated Q at ", t, b_idx[0], b_idx[1], e_idx, " is ", Q_i[run, t, b_idx[0], b_idx[1], e_idx])

                logging = False
                if logging:
                    print("The non zero values of Q: ", np.nonzero(Q_i))

                belief = next_belief
                b_idx[0] = int(belief[0] * belief_res) if belief[0] < 1 else belief_res - 1
                b_idx[1] = int(belief[1] * belief_res) if belief[1] < 1 else belief_res - 1

                state_vec = Xn

    with open('Data/model_rl_nn.pkl', 'wb') as f:
        pickle.dump(Ret, f)
        # pickle.dump([Q, V, G, Ret], f)
    f.close()
    # log_file.close()
