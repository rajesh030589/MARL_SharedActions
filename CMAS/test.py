import numpy as np
from Model import SmartGrid
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    belief_res = 100
    model = SmartGrid(2, belief_res)

    belief1 = np.linspace(0, 1, belief_res)
    belief2 = np.linspace(0, 1, belief_res)
    gamma_size = 81
    R1 = []
    R2 = []
    for b1 in tqdm(range(belief_res)):
        for b2 in range(belief_res):
            belief = [belief1[b1], belief2[b2]]
            state_vec = model.random_state(belief)
            for g_idx in range(gamma_size):
                gamma = model.get_gamma(g_idx)
                _, A, _ = model.excite_model(state_vec, gamma)
                next_belief = model.Nxt_belief_vec(belief, gamma, A)
                R1.append(next_belief[0])
                R2.append(next_belief[1])

    plt.scatter(R1, R2)
    plt.show()


if __name__ == "__main__":
    main()
