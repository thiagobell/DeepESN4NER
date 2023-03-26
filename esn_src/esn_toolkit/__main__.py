import os

import torch
import numpy as np

from esn_toolkit.esn import TimeSeriesESN
from esn_toolkit.reservoir import Reservoir

import matplotlib.pyplot as plt
from p_tqdm import p_imap
from functools import partial

def generate_plot_predictions(save_path, title, targets, mean_predictions, activations: torch.Tensor, std_predictions=None):
    """
    Generates a plot comparing the predictions and targets. If std_prediction is not none, draws the standard deviation
    of the predictions onto the plot as well
    :param save_path: path to save the plot to
    :param title: title to be added to the plot
    :param targets: a list of target values
    :param mean_predictions: list of predictions (means)
    :param std_predictions: list of standard deviations (for the case of multiple predictions / trials)
    :return:
    """
    means = np.array(mean_predictions)

    time_steps = list(range(means.shape[0]))

    plt.figure()

    plt.plot(time_steps, targets, label="target")
    plt.plot(time_steps, means, label="avg prediction")
    if std_predictions is not None:
        stddevs = np.array(std_predictions)
        upper_limit = means + stddevs
        lower_limit = means - stddevs
        plt.fill_between(time_steps, lower_limit, upper_limit, color="gray", alpha=0.3, label="standard deviation")
    plt.ylabel("value")
    plt.title(title)
    plt.xlabel("time step")
    plt.ylim((-1, 1))
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    plt.figure()
    plt.imshow(activations, cmap="hot")
    plt.ylabel("steps")
    plt.xlabel("reservoir values")
    plt.savefig(save_path+"act.png")
    plt.close()



def run(data, id, reservoir_size: int, reservoir_connectivity: float, reservoir_weight_scale: float,
        input_connectivity: float, input_weight_scale: float, leaking_rate: float, lr=1e-3, weight_decay=5e-8):

    reservoir = Reservoir(1, True, reservoir_size, reservoir_connectivity, reservoir_weight_scale,
                          input_connectivity, input_weight_scale, leaking_rate, 0.79,
                          [torch.tanh, torch.sin])

    """
    res = [
        Reservoir(1, True, 100, reservoir_connectivity, reservoir_weight_scale,
                          input_connectivity, input_weight_scale, leaking_rate, 0.79,
                          [torch.tanh, torch.sin]),
        Reservoir(100, False, 300, reservoir_connectivity, reservoir_weight_scale,
                          input_connectivity, input_weight_scale, leaking_rate, 0.79,
                          [torch.tanh, torch.sin]),
        Reservoir(300, False, 500, reservoir_connectivity, reservoir_weight_scale,
                          input_connectivity, input_weight_scale, leaking_rate, 0.79,
                          [torch.tanh, torch.sin]),
        Reservoir(500, True, 100, reservoir_connectivity, reservoir_weight_scale,
                          input_connectivity, input_weight_scale, leaking_rate, 0.79,
                          [torch.tanh, torch.sin])
    ]
    reservoir = DeepReservoir(res)
    """

    training_size = 9700
    test_size = 300
    epochs = 150
    initial_transient = 300
    # for epochs in range(5):
    esn = TimeSeriesESN(reservoir, 1)

    esn.train_readout(epochs, data[:training_size], initial_transient, lr=lr, weight_decay=weight_decay)
    # must include the last training value to work as a "seed"

    predictions, activations, loss = esn.evaluate_recursive(data[training_size - 1:training_size + test_size, :])
    targets = data[training_size:, :]
    experiments_folder = f"/home/tbellfelix/results/{id:0>5d}"
    os.mkdir(experiments_folder)
    generate_plot_predictions(experiments_folder+f"/{id}_result{epochs}_test.png", "title", targets[0:test_size], predictions[0:test_size],
                              activations)

    # predict on the training set
    esn.warmup(data[0:initial_transient])
    predictions, activations, loss = esn.evaluate_recursive(data[initial_transient: training_size, :])
    targets = data[initial_transient+1:training_size, :]
    generate_plot_predictions(experiments_folder+f"/{id}_result{epochs}_train.png", "title", targets[0:400],
                              predictions[0:400], activations)
    esn.reservoir.reset_reservoir_state()
    return loss


def gridsearch():
    data = torch.from_numpy(np.loadtxt(os.path.join(os.path.dirname(__file__), "../../data/MackeyGlass_t17.txt")))
    data = data.reshape(-1, 1).float()

    param_settings = {
        "reservoir_size": [300, 500],
        "reservoir_connectivity": [0.1, 0.5, 1.0],
        "reservoir_weight_scale": [0.1, 0.5, 1.2],
        "input_connectivity": [0.1, 0.5, 1.0],
        "input_weight_scale": [0.1, 0.5, 1.2],
        "leaking_rate": [0.3, 0.7, 0.9, 0.99],
        "lr": [1e-2, 1e-3, 1e-4, 1e-5],
        "weight_decay": [5e-5, 5e-6, 5e-7, 5e-8, 5e-9]
    }
    params = []

    for key in param_settings.keys():
        if len(params) == 0:
            params = [{key: value} for value in param_settings[key]]
        else:
            new_list = []
            for param in params:
                for value in param_settings[key]:
                    p_copy = param.copy()
                    p_copy[key] = value
                    new_list.append(p_copy)
            params = new_list

    # adding ids
    for inx, _ in enumerate(params):
        params[inx]["id"] = inx

    def run_exp(par):
        return par["id"], run(data, **par)

    min_loss = 100000
    min_loss_id = None

    for id, loss in p_imap(run_exp, params, num_cpus=4):
        if loss < min_loss:
            min_loss = loss
            min_loss_id = id
        params[id]["loss"] = loss
    with open("/home/tbellfelix/results/results.json", "w") as f:
        import json

        json.dump(params, f)
    print(f"min loss{min_loss}")
    print(f"min loss id{min_loss_id}")


if __name__ == "__main__":
    #data = torch.from_numpy(np.loadtxt(os.path.join(os.path.dirname(__file__), "../../data/MackeyGlass_t17.txt")))
    #data = data.reshape(-1, 1).float()
    #run(data, "single_exp", 500, 0.5, 0.1, 0.1, 1.2, 0.3)
    gridsearch()

