import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
from benchmark_systems import load_data
from helpers_models import build_RC_model
from cpsmehelper import export_figure

from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.pruning import NetworkPruner


# Add the directory containing the style file to the matplotlib style path
plt.style.use(os.path.join(os.getcwd(), "AIP_journal.mplstyle"))
FIGURE_PATH = os.path.join(os.getcwd(), "figures")

"""
NARMA-5 system 
input: [x(t)], 
output: [x(t + 1)]
"""

# load data
x_train, y_train, x_test, y_test = load_data(name="narma5", n_samples=10)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

fig = plt.figure()
plt.plot(x_train[0, :, 0], label=r"$x(t)$")
plt.legend()
plt.xlabel("time")
plt.ylabel("state")
plt.title("NARMA-5 system")
plt.tight_layout()
export_figure(
    fig=fig,
    name="narma5_trajectories.pdf",
    savedir=FIGURE_PATH,
    style="presentation_2x2",
)
# plt.show()


"""
RC-based modeling
"""

# define properties of the RC as dictionary
rc_config = {
    "nodes": 200,  # number of reservoir nodes
    "density": 0.5,  # connection density in reservoir
    "activation": "tanh",  # activation function
    "fraction_input": 0.5,  # fraction of input-receiving nodes
    "fraction_output": 1.0,  # fraction of read-out nodes
    "metric": "mse",  # mean squared error metric
    "transients": 50,  # discard the first 50 time steps
}

# define graph analyzer that will extract properties of the reservoir
graph_analyzer = GraphAnalyzer()

# define RC Pruner
pruner = NetworkPruner(
    stop_at_minimum=True,
    min_num_nodes=2,
    patience=5,
    candidate_fraction=0.25,
    remove_isolated_nodes=False,
    metrics=["mse"],
    maintain_spectral_radius=False,
    return_best_model=True,
)


# data shapes
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:]

"""
Loop over many RC models for the same parameters and the same data:
- record scores
- record properties of the reservoir
"""

num_trials = 100

# store models, scores, properties
models, scores, graph_props = [], [], []
models_pruned, scores_pruned, graph_props_pruned = [], [], []
pruning_histories = []


for i in range(num_trials):

    # Classical RC modeling using a random reservoir graph

    _model = build_RC_model(
        input_shape=input_shape, output_shape=output_shape, configuration=rc_config
    )

    # train the model
    _model.fit(x=x_train, y=y_train)

    # evaluate some metrics on test set
    _score = _model.evaluate(x=x_test, y=y_test)[0]
    print(f"trial {i+1}/{num_trials}: test set score:\t{_score:.4f}")

    # extract graph-level properties of the reservoir
    _graph_props = graph_analyzer.extract_properties(_model.reservoir_layer.weights)

    # # print truth and predicted sequence
    # # make predictions
    # y_pred = _model.predict(x_test)

    # plt.figure()
    # plt.plot(
    #     y_test[0, rc_config["transients"] :, 0],
    #     label="ground truth",
    #     marker=".",
    #     color="#1D3557",
    # )
    # plt.plot(y_pred[0, :, 0], label="prediction", marker=".", color="#E63946")
    # plt.legend()
    # plt.xlabel("time")
    # plt.title(
    #     rf"Test set predictions for $\sin(t) \mapsto \cos(t)$, MSE={metric_value[0]:.4f}"
    # )
    # plt.show()

    ## extract properties of the reservoir
    ## ToDo: import the GraphProps Extractor class
    # rc_graph_props = model.get_reservoir_properties()

    # store the model, scores, and properties
    models.append(_model)
    scores.append(_score)
    graph_props.append(_graph_props)

    # Pruning the random reservoir
    _model_pruned, _history = pruner.prune(
        model=_model, data_train=(x_train, y_train), data_val=(x_test, y_test)
    )

    print(f"took {_history['iteration'][-1]+1} iterations to prune the model")
    for key in _history["graph_props"].keys():
        print(
            f"{key}: \t initial model {_history['graph_props'][key][0]:.4f}; \t final model: {_history['graph_props'][key][-1]:.4f}"
        )

    # evaluate some metrics on test set
    _score_pruned = _model_pruned.evaluate(x=x_test, y=y_test)[0]

    # extract graph-level properties of the reservoir
    _graph_props_pruned = graph_analyzer.extract_properties(
        _model_pruned.reservoir_layer.weights
    )

    models_pruned.append(_model_pruned)
    scores_pruned.append(_score_pruned)
    graph_props_pruned.append(_graph_props_pruned)
    pruning_histories.append(_history)

    del (
        _score,
        _score_pruned,
        _model,
        _model_pruned,
        _graph_props,
        _graph_props_pruned,
        _history,
    )


scores = np.array(scores)
scores_pruned = np.array(scores_pruned)


"""
Comparison of pruned RC with the classical model
"""

# 1. variation of the model scores (i.e. effect of random aspects in the modeling)
# plt.figure()
# plt.hist(scores, bins=20, color="#457B9D", density=True, label="random RC")
# # plt.hist(scores_pruned, bins=20, color='red', density=True, label="pruned RC")
# plt.xlabel("MSE")
# plt.ylabel("probability")
# plt.title("Test set scores")
# plt.show()


# todo: truncate violins at zero
plt.figure()  # Violin plot of the scores array
sns.violinplot(data=scores, color="#457B9D")
plt.xlabel("Model")
plt.ylabel("MSE")
plt.title("Distribution of test set scores")
plt.savefig("random_model_scores.pdf")
plt.show()

# 2. Improvements in the scores due to pruning

# plt.figure()
# sns.violinplot(data=scores_pruned - scores, color="#457B9D")
# plt.xlabel("Model")
# plt.ylabel("MSE")
# plt.title("Improvements of test set scores")
# plt.show()
