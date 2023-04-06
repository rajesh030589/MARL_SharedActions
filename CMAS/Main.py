from BR import model_seq, model_rnd
from Plot_file import plot_figure
from RL import model_rl, modelfree_rl, model_rl_nn
from tqdm import tqdm


def main():

    belief_res = 10
    TimeI = 10
    EpisodeI = 10000

    # # # Model with Sequential decomposition
    # RunsI = 1
    # model_seq(RunsI, belief_res, TimeI)

    RunsI = 1
    model_rl(RunsI, EpisodeI, belief_res, TimeI)

    # # # model_rl_nn(RunsI, belief_res, TimeI)

    RunsI = 1
    Filter = 500
    modelfree_rl(RunsI, EpisodeI, belief_res, TimeI, Filter)

    # # modelfree_rl(RunsI, belief_res, TimeI, 2000)

    # RunsI = 1
    # Filter = 20
    # modelfree_rl(RunsI, EpisodeI, belief_res, TimeI, Filter)

    RunsI = 1
    Filter = 100
    modelfree_rl(RunsI, EpisodeI, belief_res, TimeI, Filter)

    plot_figure()


if __name__ == "__main__":
    main()
