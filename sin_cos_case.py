import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from benchmark_systems import load_data
from helpers_models import build_RC_model

# Generate a simple case for the sin-cos system: x: sin(t), y: cos(t)

# load data
x_train, y_train, x_test, y_test = load_data(name="sincos", n_samples=10)

sample_idx = 0
plt.figure()
plt.subplot(2, 1, 1)
for i in range(x_train.shape[2]):
    plt.plot(x_train[sample_idx, :, i], label="x_train")
plt.legend()
plt.title(r"$\sin(t) \mapsto \cos(t)$")

plt.subplot(2, 1, 2)
for i in range(y_train.shape[2]):
    plt.plot(y_train[sample_idx, :, i], label="y_train")
plt.legend()
plt.xlabel(r"$t$")
plt.show()


"""
RC-based modeling
"""

# define properties of the RC as dictionary
rc_config = {
    "nodes": 200,  # number of reservoir nodes
    "density": 0.5,  # connection density
    "activation": "tanh",  # activation function
    "fraction_input": 0.5,  # fraction of input-receiving nodes
    "fraction_output": 1.0,  # fraction of read-out nodes
    "metric": "mse",  # mean squared error metric
    "transients": 50,  # discard the first 50 time steps
}

# data shapes
input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:]

"""
Loop over many RC models for the same parameters and the same data
"""

num_trials = 10000

# store models, scores, properties
models, scores, model_props = [], [], []
models_pruned, scores_pruned, model_props_pruned = [], [], []

for i in range(num_trials):

    """
    Classical "random" RC modeling
    """

    model = build_RC_model(
        input_shape=input_shape, output_shape=output_shape, configuration=rc_config
    )

    # train the model
    model.fit(x_train, y_train)

    # make predictions
    y_pred = model.predict(x_test)

    # evaluate some metrics on test set
    metric_value = model.evaluate(X=x_test, y=y_test)
    print(f"trial {i+1}/{num_trials}: test set score:\t{metric_value[0]:.4f}")

    # # print truth and predicted sequence
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
    models.append(model)
    scores.append(metric_value[0])
    # model_props.append(rc_graph_props)

    """
    Pruning the random reservoir
    """

    # create a RC Pruner object

    # Prune the RC

    # score_pruned.append()

    # Extract RC properties


scores = np.array(scores)
# scores_pruned = np.array(scores_pruned)


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
