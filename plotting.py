from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import os
from cpsmehelper import export_figure

# Add the directory containing the style file to the matplotlib style path
plt.style.use(os.path.join(os.getcwd(), "AIP_journal.mplstyle"))
FIGURE_PATH = os.path.join(os.getcwd(), "figures")


def plot_loss_histories(
    histories: list,
    x_quantity: str = "nodes",
    color: str = "gray",
    display_trial_num: bool = True,
):

    if x_quantity == "nodes":
        key_x = "num_nodes"
        x_label = "nodes"
    elif x_quantity == "iterations":
        key_x = "iterations"
        x_label = "iterations"
    else:
        raise ValueError(f"Unknown x_quantity: {x_quantity}")

    fig = plt.figure()
    for history in histories:
        # plot complete iteration history
        plt.plot(
            history[key_x],
            history["loss"],
            color=color,
            alpha=0.5,
        )

        # plot the starting point
        plt.plot(
            history["num_nodes"][0],
            history["loss"][0],
            color=color,
            alpha=0.8,
            marker=".",
            markersize=10,
            linestyle="None",
        )

        # find minimum value in loss
        idx_min = np.argmin(history["loss"])

        # place a marker at the minimum value
        plt.plot(
            history["num_nodes"][idx_min],
            history["loss"][idx_min],
            color=color,
            alpha=0.8,
            marker="x",
            markersize=6,
            linestyle="None",
        )

        # plot a text box that reports the number of trials
    if display_trial_num:
        num_trials = len(histories)
        plt.text(
            0.75,
            1.03,
            f"# trials: {num_trials}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
        )
    plt.xlabel(x_label)
    plt.ylabel("loss")

    return fig


def plot_property_histories(
    histories: list,
    x_quantity="nodes",
    y_quantity: list = "density",
    color: str = "gray",
    display_trial_num: bool = True,
):

    if x_quantity == "nodes":
        key_x = "num_nodes"
        x_label = "nodes"
    elif x_quantity == "iterations":
        key_x = "iterations"
        x_label = "iterations"
    else:
        raise ValueError(f"Unknown x_quantity: {x_quantity}")

    fig = plt.figure()
    for history in histories:
        # plot complete iteration history
        plt.plot(
            history[key_x],
            history["graph_props"][y_quantity],
            color=color,
            alpha=0.5,
        )

        # plot the starting point
        plt.plot(
            history["num_nodes"][0],
            history["graph_props"][y_quantity][0],
            color=color,
            alpha=0.8,
            marker=".",
            markersize=10,
            linestyle="None",
        )

        # find minimum value in loss
        idx_min = np.argmin(history["loss"])

        # place a marker at the final model
        plt.plot(
            history["num_nodes"][idx_min],
            history["graph_props"][y_quantity][idx_min],
            color=color,
            alpha=0.8,
            marker="x",
            markersize=6,
            linestyle="None",
        )

    # plot a text box that reports the number of trials
    if display_trial_num:
        num_trials = len(histories)
        plt.text(
            0.75,
            1.03,
            f"# trials: {num_trials}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
        )

    plt.xlabel(x_label)
    plt.ylabel(y_quantity.replace("_", " "))

    return fig


def plot_violin_comparison(
    x1,
    x2,
    labels: list,
    quantity: str,
    cut_zero=True,
    display_trial_num: bool = True,
):

    # whether to bound the violin plot at zero
    if cut_zero:
        cut = 0
    else:
        cut = None

    fig = plt.figure()
    sns.violinplot(
        data=[x1, x2],
        palette=["#457B9D", "#E63946"],
        inner="quartiles",
        log_scale=False,
        cut=cut,
        dodge=False,
    )
    # plot a text box that reports the number of trials
    if display_trial_num:
        num_trials = len(x1)
        plt.text(
            0.75,
            1.03,
            f"# trials: {num_trials}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
        )
    plt.ylabel(quantity.replace("_", " "))
    plt.xticks([0, 1], labels)

    return fig
