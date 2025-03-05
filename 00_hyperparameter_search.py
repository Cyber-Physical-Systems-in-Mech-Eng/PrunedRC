import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
from benchmark_systems import load_data

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.cross_validation import cross_val

from helpers_models import build_RC_model

# load data
x_train, y_train, x_test, y_test = load_data(name="narma5", n_samples=10)
data_train = (x_train, y_train)
data_test = (x_test, y_test)

input_shape, output_shape = x_train.shape[1:], y_train.shape[1:]

# define hyperparameters and their search space
hp = {
    "nodes": np.arange(50, 201, 25).astype(int),
    "leakage_rate": np.arange(0.1, 1, 0.1),
    "density": np.arange(0.1, 0.8, 0.1),
    "activation": ["tanh", "sigmoid"],
    "fraction_input": [0.5],
    "fraction_output": [0.5],
    "metric": ["mse"],
    "transients": [50],
}

# create search grid: all possible combinations of hyperparameters
# Generate all possible combinations
keys, values = zip(*hp.items())
hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Number of hyperparameter combinations: {len(hp_combinations)}")

# loop over all possible combinations and obtain cross-validated model score
for i, _config in enumerate(hp_combinations):
    print(f"Combination {i+1}/{len(hp_combinations)}: {_config}")

    # build reservoir computer
    _model = build_RC_model(input_shape, output_shape, configuration=_config)

    # cross-validate RC
    _val, _mean, _std_dev = cross_val(
        _model, x_train, y_train, n_splits=5, metric=["mse"]
    )

    # update hyperparameter dictionary
    hp_combinations[i]["mean_score"] = _mean
    hp_combinations[i]["std_score"] = _std_dev

    print(f"mean score: \t {_mean:.4f}, std dev: \t{_std_dev:.4f}\n")


# save results to file
with open("hyperparameter_search_narma5.pkl", "wb") as f:
    pickle.dump(hp_combinations, f)

# find best hyperparameters
best_idx = np.argmin(mean_scores)

print(f"\n\nBest hyperparameters: {hp_combinations[best_idx]}\n\n")


# some analysis of the results - are there global trends?
node_list = []
score_list = []
leakage_list = []
density_list = []

for _config in hp_combinations:
    node_list.append(_config["nodes"])
    score_list.append(_config["mean_score"])
    leakage_list.append(_config["leakage_rate"])
    density_list.append(_config["density"])

# Determine the y-axis limits
y_min = min(score_list)
y_max = max(score_list)


plt.figure()
plt.subplot(1, 3, 1)
plt.scatter(node_list, score_list)
plt.xlabel("Nodes")
plt.ylabel("Mean loss")
plt.ylim(y_min, y_max)

plt.subplot(1, 3, 2)
plt.scatter(leakage_list, score_list)
plt.xlabel("Leakage Rate")
plt.ylabel("Mean loss")
plt.title(f"best model: {hp_combinations[best_idx]}")
plt.ylim(y_min, y_max)

plt.subplot(1, 3, 3)
plt.scatter(density_list, score_list)
plt.xlabel("Density")
plt.ylabel("Mean loss")
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.savefig("hyperparameter_search_narma5.png")
plt.show()
