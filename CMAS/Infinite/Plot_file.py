from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from matplotlib import colors as mcolors


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_figure():

    # colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # print(colors)

    f = open('/home/raj/Dropbox/MARL/Python Codes/Infinite/Data/model_seq.pkl', 'rb')
    [Ret_model_seq, G_model_seq] = pickle.load(f)
    f.close()

    f = open('/home/raj/Dropbox/MARL/Python Codes/Infinite/Data/model_rl.pkl', 'rb')
    [Ret_model_rl, G_model_rl] = pickle.load(f)
    f.close()

    f = open('/home/raj/Dropbox/MARL/Python Codes/Infinite/Data/modelfree_500_rl.pkl', 'rb')
    Ret_modelfree1_rl = pickle.load(f)
    f.close()

    f = open('/home/raj/Dropbox/MARL/Python Codes/Infinite/Data/modelfree_20_rl.pkl', 'rb')
    Ret_modelfree4_rl = pickle.load(f)
    f.close()

    # f = open('/home/raj/Dropbox/MARL/Python Codes/Final_Codes/Data/modelfree_5000_rl.pkl', 'rb')
    # G_modelfree3_rl = pickle.load(f)
    # f.close()

    f = open('/home/raj/Dropbox/MARL/Python Codes/Infinite/Data/modelfree_100_rl.pkl', 'rb')
    Ret_modelfree3_rl = pickle.load(f)
    f.close()

    Ret_model_seq = np.mean(Ret_model_seq)
    Ret_mod_seq = Ret_model_seq*np.ones(len(Ret_model_rl))

    # Figures we need:
    # 1) Returns plot i.e. Value function at t = 0 for all episodes with model free and seq vs RL

    plt.figure()
    # plt.plot(Ret_model_rl[:, 0], label='Model RL', color='#87CEEB')
    # plt.plot(Ret_modelfree1_rl[:, 0], label='Model RL', color='lightgreen')
    # plt.plot(Ret_modelfree3_rl[:, 0], label='Model RL', color='lightsalmon')
    plt.plot(Ret_mod_seq, '--', label='Model SDBR')
    plt.plot(smooth(Ret_model_rl, 1000)[:-500], label='Model RL', color='b')
    plt.plot(smooth(Ret_modelfree1_rl, 1000)[:-500], label='Model Free RL K = 500', color='g')
    plt.plot(smooth(Ret_modelfree3_rl, 1000)[:-500], label='Model Free RL K = 100', color='r')
    plt.plot(smooth(Ret_modelfree4_rl, 1000)[:-500], label='Model Free RL K = 20', color='k')
    # plt.plot(Ret_modelfree1_rl[:, 0], label='Model Free RL, K = 500')
    # plt.plot(Ret_modelfree3_rl[:, 0], label='Model Free RL, K = 5000')
    plt.xlabel('Time Iteration')
    plt.ylabel('Rewards')
    plt.xlim([0, len(Ret_model_rl)])
    plt.ylim([-5, 0])
    plt.legend(loc='best')
    plt.grid(True)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.savefig('/home/raj/Dropbox/MARL/Python Codes/Infinite/Figures/Ret_episodes.pdf')

    # 2) Value function plot from t = 0 to T i.e. V(0) to V(T) with model free and seq vs RL vs random policy

    # plt.figure()
    # plt.plot(Ret_model_seq, label='Model SDBR')
    # plt.plot(np.mean(Ret_model_rl[-100:, :], axis=0), label='Model RL')
    # plt.plot(np.mean(Ret_modelfree1_rl[-100:, :], axis=0), label='Model Free RL')
    # plt.plot(np.mean(Ret_modelfree3_rl[-100:, :], axis=0), label='Model Free RL')
    # plt.plot(np.mean(Ret_modelfree4_rl[-100:, :], axis=0), label='Model Free RL')
    # plt.xlabel('Time')
    # plt.ylabel('Value Function')
    # plt.xlim([0, 10])
    # plt.ylim([-5, 0])
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.savefig('/home/raj/Dropbox/MARL/Python Codes/Final_Codes/Figures/Value_function.pdf')

    # plt.figure()
    # plt.plot(Ret_mod_seq, label='Model SDBR')
    # plt.plot(Ret_model_rl[:, 0], label='Model RL')
    # plt.plot(Ret_modelfree1_rl[:, 0], label='Model Free RL, K = 500')
    # plt.plot(Ret_modelfree3_rl[:, 0], label='Model Free RL, K = 5000')
    # plt.xlabel('Num of Episodes')
    # plt.ylabel('Rewards')
    # plt.xlim([0, len(Ret_model_rl)])
    # plt.ylim([-5, 0])
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.savefig('/home/raj/Dropbox/MARL/Python Codes/Final_Codes/Figures/Ret_episodes.pdf')

#     # # 3) Policy plot i.e. Value function at t = 0 for all episodes with model free and seq vs RL

#     plt.figure()
#     plt.pcolor(G_model_seq[:, :, 0, 1])
#     plt.figure()
#     plt.pcolor(G_model_rl[:, :, 0, 1])
# #     # plt.figure()
    plt.show()
#     plt.clf()
#     # plt.pcolor(G_modelfree1_rl[:, :, 0, 1])
#     # plt.figure()
#     # plt.pcolor(G_modelfree3_rl[:, :, 0, 1])

#     # for j in range(50):
#     #     R = np.zeros(12000)
#     #     C = np.zeros(12000)
#     #     for t in range(12000):
#     #         R[t] = np.sum(Ret_model_rl[t, j, :, :])
#     #         C[t] = np.count_nonzero(Ret_model_rl[t, j, :, :])
#     #         if C[t] == 0:
#     #             R[t] = 0
#     #         else:
#     #             R[t] = R[t]/C[t]
#     #     Ret_mod_rl = R  # np.squeeze(np.apply_over_axes(np.mean(np.nonzero), Ret_model_rl, [1, 2]))
#     #     Ret_mod_seq = V_model_seq[j]*np.ones(len(Ret_mod_rl))
#     #     plt.figure()
#     #     plt.plot(Ret_mod_rl, label='Model RL')
#     #     plt.plot(Ret_mod_seq, label='Model SDBR')
#     # plt.show()
#     # j = 44
#     # R = np.zeros(12000)
#     # C = np.zeros(12000)
#     # for t in range(12000):
#     #     R[t] = np.sum(Ret_model_rl[t, j, :, :])
#     #     C[t] = np.count_nonzero(Ret_model_rl[t, j, :, :])
#     #     if C[t] == 0:
#     #         R[t] = 0
#     #     else:
#     #         R[t] = R[t]/C[t]
#     # Ret_mod_rl = R  # np.squeeze(np.apply_over_axes(np.mean(np.nonzero), Ret_model_rl, [1, 2]))
#     # plt.figure()
#     # plt.plot(Ret_mod_rl, label='Model RL')
#     # plt.plot(Ret_mod_seq, label='Model SDBR')
#     # plt.show()

#     # Ret_model_rl = R  # np.squeeze(np.apply_over_axes(np.mean(np.nonzero), Ret_model_rl, [1, 2]))
#     # Ret_modelfree1_rl = np.squeeze(np.apply_over_axes(np.mean, Ret_modelfree1_rl, [1, 2]))
#     # Ret_modelfree2_rl = np.squeeze(np.apply_over_axes(np.mean, Ret_modelfree2_rl, [1, 2]))
#     # Ret_modelfree3_rl = np.squeeze(np.apply_over_axes(np.mean, Ret_modelfree3_rl, [1, 2]))
#     # Ret_model_seq = V_model_seq[48]*np.ones(len(Ret_model_rl))
#     # # Ret_modelfree_seq = -V_modelfree_seq[-1]*np.ones(len(Ret_modelfree_rl))
#     # # Ret_modelfree_seq = V_modelfree_seq[-1]*np.ones(len(Ret_modelfree_rl))


#     # plt.show()
#     # # plt.clf()
#     # V_model_rnd = np.flipud(np.squeeze(np.apply_over_axes(np.mean, V_model_rnd, [1, 2])))
#     # V_model_rl = np.squeeze(np.apply_over_axes(np.mean, V_model_rl, [1, 2]))
#     # V = V_model_rl
#     # V = np.zeros(50)
#     # for t in range(50):
#     #     V[t] = np.sum(V_model_rl[t, :, :])/np.count_nonzero(V_model_rl[t, :, :])
