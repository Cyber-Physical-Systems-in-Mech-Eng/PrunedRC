import os

from pyreco.pruning import NetworkPruner
from pyreco.graph_analyzer import GraphAnalyzer

from benchmark_systems import load_data
from helpers_models import run_pruning_trials


# Specify name of the file to store the results
SAVE_NAME = "narma5.pkl"
SAVE_PATH = os.path.join(os.getcwd(), "stored_results", SAVE_NAME)
NUM_TRIALS = 10  # how many models to build and prune

if __name__ == "__main__":
    # NARMA-5 system
    # input: [x(t)],
    # output: [x(t + 1)]

    # load data
    x_train, y_train, x_test, y_test = load_data(name="narma5", n_samples=10)
    data_train = (x_train, y_train)
    data_test = (x_test, y_test)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Define RC properties for the models that will later be pruned
    rc_config = {
        "nodes": 50,  # number of reservoir nodes
        "density": 0.5,  # connection density in reservoir
        "activation": "tanh",  # activation function
        "fraction_input": 0.5,  # fraction of input-receiving nodes
        "fraction_output": 0.8,  # fraction of read-out nodes
        "metric": "mse",  # mean squared error metric
        "transients": 50,  # discard the first 50 time steps
        "leakage_rate": 0.1,  # leakage rate of the reservoir
    }

    # define graph analyzer that will extract properties of the reservoir
    graph_analyzer = GraphAnalyzer()

    # define RC network pruner and relevant properties
    pruner = NetworkPruner(
        stop_at_minimum=True,
        min_num_nodes=2,
        patience=3,
        candidate_fraction=0.25,
        metrics=["mse"],
        return_best_model=True,  # should always be True
        graph_analyzer=graph_analyzer,
    )

    # run the pruning trials (this takes a while). Stores results to disk
    run_pruning_trials(
        data_train=data_train,
        data_test=data_test,
        rc_config=rc_config,
        graph_analyzer=graph_analyzer,
        network_pruner=pruner,
        num_trials=NUM_TRIALS,
        save_path=SAVE_PATH,
    )
