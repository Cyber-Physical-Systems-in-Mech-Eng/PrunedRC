"""
Cross-validated hyperparameter grid-search for the RC model.

Should be run from command line with the following command:
python 00_hyperparameter_search.py --case narma5
"""

import numpy as np
import itertools
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyreco.cross_validation import cross_val
from helpers_models import build_RC_model

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Hyperparameter search for RC model")
parser.add_argument("--case", type=str, required=True, help="Case study name")
args = parser.parse_args()

# define the case study
CASE = args.case

# whether to run the hyperparameter study or just analyse results
RUN_HYPERPARAMETER_SEARCH = True

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


if RUN_HYPERPARAMETER_SEARCH:
    # define hyperparameters and their search space
    hp = {
        "nodes": np.arange(25, 101, 25).astype(int),
        "leakage_rate": np.arange(0.1, 0.951, 0.05),
        "density": np.arange(0.05, 0.251, 0.1),
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
    mean_scores = []
    rem_idx = []
    for i, _config in enumerate(hp_combinations):

        try:
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

            mean_scores.append(_mean)

            print(f"mean score: \t {_mean:.4f}, std dev: \t{_std_dev:.4f}\n")

        except Exception as e:
            print(f"Error: {e}")
            rem_idx.append(i)
            hp_combinations[i]["mean_score"] = None
            hp_combinations[i]["std_score"] = None

    # in case there was some issue, remove the faulty hyperparameter combination
    # delete all entries in hp_combinations that have None as mean_score
    hp_combinations = [
        hp_combinations[i]
        for i, combo in enumerate(hp_combinations)
        if combo["mean_score"] is not None
    ]

    # save results to file
    path = os.path.join(
        os.getcwd(), "stored_results", f"hyperparameter_search_{CASE}.pkl"
    )
    with open(path, "wb") as f:
        pickle.dump(hp_combinations, f)


else:  # load hyperparameter study results

    # load pickled results
    path = os.path.join(
        os.getcwd(), "stored_results", f"hyperparameter_search_{CASE}.pkl"
    )
    with open(path, "rb") as f:
        hp_combinations = pickle.load(f)

    # extract mean scores
    mean_scores = [c["mean_score"] for c in hp_combinations]


# find best set of hyperparameters
best_idx = np.argmin(mean_scores)
print("\n\nBest hyperparameters:")
for key in hp_combinations[best_idx].keys():
    print(f"{key}: \t{hp_combinations[best_idx][key]}")

# obtain top 3 hyperparameter combinations
top_3_idx = np.argsort(mean_scores)[:3]
print("\nTop 3 hyperparameters:")
for key in hp_combinations[best_idx].keys():
    print_str = []
    for idx in top_3_idx:
        print_str.append(hp_combinations[idx][key])
    print(f"{key}: \t{print_str}")


# some analysis of the results - are there global trends observable?
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
y_min = min(score_list) * 0.9
y_max = max(score_list) * 1.1

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
plt.savefig(os.path.join(os.getcwd(), "figures", f"hyperparameter_search_{CASE}.png"))
# plt.show()
plt.close()

# # 3d scatter plot with score as color code
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(node_list, leakage_list, density_list, c=score_list)
# # add colorbar
# cbar = plt.colorbar(ax.scatter(node_list, leakage_list, density_list, c=score_list))
# cbar.set_label("Mean loss")
# ax.set_xlabel("Nodes")
# ax.set_ylabel("Leakage Rate")
# ax.set_zlabel("Density")
# # plt.savefig("hyperparameter_search_narma5_3D.png")
# plt.show()


# export to LaTeX table

# remove the metric and transients keys from the hyperparameter combinations
for i, _ in enumerate(hp_combinations):
    del hp_combinations[i]["metric"]
    del hp_combinations[i]["transients"]

print("\n\nLaTeX table:\n")
# Obtain the column names from the keys of the hp combinations dictionaries
column_names = [key.replace("_", " ").lower() for key in hp_combinations[0].keys()]

# Print the column names for the LaTeX table
print(" & ".join(column_names) + " \\\\")
top_10_idx = np.argsort(mean_scores)[:10]
for idx in top_10_idx:
    print_str = []
    for key in hp_combinations[idx].keys():
        print_str.append(hp_combinations[idx][key])
    print(
        " & ".join(
            [
                (
                    f"{x:.5f}"
                    if key in ["mean_score", "std_score"]
                    else f"{x:.1f}" if isinstance(x, float) else str(x)
                )
                for key, x in zip(hp_combinations[idx].keys(), print_str)
            ]
        )
        + " \\\\"
    )
print("\n\n")

# print the latex table to a txt file
with open(
    os.path.join(os.getcwd(), "stored_results", f"hyperparameter_search_{CASE}.txt"),
    "w",
) as f:
    f.write(" & ".join(column_names) + " \\\\\n")
    for idx in top_10_idx:
        print_str = []
        for key in hp_combinations[idx].keys():
            print_str.append(hp_combinations[idx][key])
        f.write(
            " & ".join(
                [
                    (
                        f"{x:.5f}"
                        if key in ["mean_score", "std_score"]
                        else f"{x:.1f}" if isinstance(x, float) else str(x)
                    )
                    for key, x in zip(hp_combinations[idx].keys(), print_str)
                ]
            )
            + " \\\\\n"
        )
    f.write("\n\n")
