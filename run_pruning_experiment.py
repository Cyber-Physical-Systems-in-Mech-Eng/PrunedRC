import argparse
import os
import pickle
from pyreco.pruning import NetworkPruner
from pyreco.graph_analyzer import GraphAnalyzer

from helpers_models import run_pruning_trials


# Parse command line arguments
parser = argparse.ArgumentParser(description="Hyperparameter search for RC model")
parser.add_argument("--case", type=str, required=True, help="Case study name")
parser.add_argument("--trials", type=int, required=True, help="Number of trials")
args = parser.parse_args()

# define the case study
CASE = args.case
TRIALS = args.trials

SAVE_NAME = f"{CASE}_{TRIALS}_iters.pkl"
SAVE_PATH = os.path.join(os.getcwd(), "stored_results", SAVE_NAME)


# load pickled data from  local /data/ folder
data_path = os.path.join(os.getcwd(), "data", f"{CASE}_data.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)
    x_train, y_train, x_test, y_test = (
        data[0],
        data[1],
        data[2],
        data[3],
    )

data_train = (x_train, y_train)
data_test = (x_test, y_test)
input_shape, output_shape = x_train.shape[1:], y_train.shape[1:]

# Define RC properties for the models that will later be pruned
rc_config = {
    "nodes": 50,  # number of reservoir nodes
    "density": 0.1,  # connection density in reservoir
    "activation": "tanh",  # activation function
    "fraction_input": 0.5,  # fraction of input-receiving nodes
    "fraction_output": 0.5,  # fraction of read-out nodes
    "metric": "mse",  # mean squared error metric
    "transients": 50,  # discard the first 50 time steps
    "leakage_rate": 0.9,  # leakage rate of the reservoir
}

# define graph analyzer that will extract properties of the reservoir
graph_analyzer = GraphAnalyzer()

# define RC network pruner and relevant properties
pruner = NetworkPruner(
    stop_at_minimum=True,
    min_num_nodes=15,
    patience=5,
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
    num_trials=TRIALS,
    save_path=SAVE_PATH,
)
