import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
import pickle

from cpsmehelper import export_figure
from plotting import (
    plot_loss_histories,
    plot_property_histories,
    plot_violin_comparison,
)


# Add the directory containing the style file to the matplotlib style path
plt.style.use(os.path.join(os.getcwd(), "AIP_journal.mplstyle"))
FIGURE_PATH = os.path.join(os.getcwd(), "figures")

# specify the path to the stored results (from corresponding ... _compute.py function)
DATA_FILE = os.path.join(os.getcwd(), "stored_results", "narma5.pkl")

if __name__ == "__main__":

    # load data
    with open(DATA_FILE, "rb") as file:
        data = pickle.load(file)

    # Extract the individual components from the loaded data
    data_train = data["data_train"]
    data_test = data["data_test"]
    rc_config = data["rc_config"]
    models = data["models"]
    scores = data["scores"]
    graph_props = data["graph_props"]
    models_pruned = data["models_pruned"]
    scores_pruned = data["scores_pruned"]
    graph_props_pruned = data["graph_props_pruned"]
    pruning_histories = data["pruning_histories"]

    """
    ANALYSIS OF THE PRUNING RESULTS. 

    A: Pruning histories (quantities along pruning iterations)
    """
    # loss vs. number of nodes (we could also place iterations as x-axis)
    fig = plot_loss_histories(pruning_histories, x_quantity="nodes")
    export_figure(
        fig=fig,
        name="narma5_loss_histories.pdf",
        savedir=FIGURE_PATH,
        style="presentation_2x3",
    )
    plt.show()

    # plot network properties vs. number of nodes (or iterations)
    properties = [
        "density",
        "av_out_degree",
        "av_in_degree",
        "clustering_coefficient",
        "spectral_radius",
    ]
    for prop in properties:
        fig = plot_property_histories(
            pruning_histories, x_quantity="nodes", y_quantity=prop
        )
        export_figure(
            fig=fig,
            name=f"narma5_{prop}_histories.pdf",
            savedir=FIGURE_PATH,
            style="presentation_2x3",
        )
    plt.show()

    """
    B: Comparison of pruned RC with the classical model
    """
    # model scores (beginning and end of pruning)
    scores = np.array(scores)
    scores_pruned = np.array(scores_pruned)

    # Violin plot for the scores and pruned scores
    fig = plot_violin_comparison(
        scores,
        scores_pruned,
        labels=["classic RC", "pruned RC"],
        quantity="loss",
    )
    export_figure(
        fig=fig,
        name="narma5_scores_comparison.pdf",
        savedir=FIGURE_PATH,
        style="presentation_2x3",
    )
    plt.show()

    # plot network properties comparing initial and pruned models as violin plots
    properties = [
        "density",
        "av_out_degree",
        "av_in_degree",
        "clustering_coefficient",
        "spectral_radius",
    ]
    for prop in properties:

        # extract the properties
        prop_initial, prop_pruned = [], []
        for history in pruning_histories:
            prop_initial.append(history["graph_props"][prop][0])
            prop_pruned.append(history["graph_props"][prop][-1])

        fig = plot_violin_comparison(
            prop_initial,
            prop_pruned,
            labels=["classic RC", "pruned RC"],
            quantity=prop,
        )

        export_figure(
            fig=fig,
            name=f"narma5_{prop}_comparison.pdf",
            savedir=FIGURE_PATH,
            style="presentation_2x3",
        )
    plt.show()

    """
    C: Statistics of pruned nodes
    """

    # was the pruned node an input-receiving / output sending one?
    is_input = []
    is_output = []
    for history in pruning_histories:
        idx_min = np.argmin(history["loss"])

        is_input.append(history["del_node_props"]["input_receiving_node"][:idx_min])
        is_output.append(history["del_node_props"]["output_sending_node"][:idx_min])

    is_input = np.array([item for sublist in is_input for item in sublist]).astype(int)
    is_output = np.array([item for sublist in is_output for item in sublist]).astype(
        int
    )

    # Count the occurrences of 0 and 1
    input_counts = np.bincount(is_input) / len(is_input)
    output_counts = np.bincount(is_output) / len(is_output)

    # Plot bar plot for input-receiving nodes
    fig = plt.figure()
    plt.bar([0, 1], input_counts, color="#457B9D", alpha=0.5, label="input nodes")
    plt.xticks([0, 1], ["no", "yes"])
    plt.xlabel("pruned node was input receiving")
    plt.ylabel("fraction")
    # plt.legend()
    plt.show()

    # Plot bar plot for output-sending nodes
    fig = plt.figure()
    plt.bar([0, 1], output_counts, color="#E63946", alpha=0.5, label="output nodes")
    plt.xticks([0, 1], ["no", "yes"])
    plt.xlabel("pruned node was output sending")
    plt.ylabel("fraction")
    # plt.legend()
    plt.show()
