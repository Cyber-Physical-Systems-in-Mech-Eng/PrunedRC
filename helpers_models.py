from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.graph_analyzer import GraphAnalyzer
from pyreco.pruning import NetworkPruner
import time
import pickle
import copy


def run_pruning_trials(
    data_train,
    data_test,
    rc_config,
    graph_analyzer,
    network_pruner,
    num_trials,
    save_path,
):
    """
    Runs multiple trials of building and pruning RC models.
    Store the models, scores, and properties of the reservoirs to local pickle file
    for later analysis.
    """

    # store models, scores, properties
    models, scores, graph_props = [], [], []
    models_pruned, scores_pruned, graph_props_pruned = [], [], []
    pruning_histories = []

    start_time = time.time()
    for i in range(num_trials):

        # Classical RC modeling using a random reservoir graph
        _model, _score, _graph_props = fit_evaluate_model(
            data_train,
            data_test,
            rc_config,
            graph_analyzer,
        )

        # store the model, scores, and properties
        models.append(_model)
        scores.append(_score)
        graph_props.append(_graph_props)
        print(f"trial {i+1}/{num_trials}: test set score:\t{_score:.4f}")

        # Pruning the random reservoir
        pruner = copy.deepcopy(network_pruner)  # we need to reset the pruner
        _model_pruned, _score_pruned, _graph_props_pruned, _history = (
            prune_evaluate_model(
                _model,
                data_train,
                data_test,
                pruner,
                graph_analyzer,
            )
        )

        # store the model, scores, properties, and pruning history
        models_pruned.append(_model_pruned)
        scores_pruned.append(_score_pruned)
        graph_props_pruned.append(_graph_props_pruned)
        pruning_histories.append(_history)

        # store intermediate results to temporary file
        with open(save_path, "wb") as temp_file:
            pickle.dump(
                {
                    "data_train": data_train,
                    "data_test": data_test,
                    "rc_config": rc_config,
                    "models": models,
                    "scores": scores,
                    "graph_props": graph_props,
                    "models_pruned": models_pruned,
                    "scores_pruned": scores_pruned,
                    "graph_props_pruned": graph_props_pruned,
                    "pruning_histories": pruning_histories,
                },
                temp_file,
            )

        del (
            _score,
            _score_pruned,
            _model,
            _model_pruned,
            _graph_props,
            _graph_props_pruned,
            _history,
        )

    print(f"total time elapsed: {time.time() - start_time:.2f} seconds")
    print("done")

    # return models, scores, graph_props, models_pruned, scores_pruned, graph_props_pruned, pruning_histories


def build_RC_model(input_shape, output_shape, configuration: dict) -> RC:
    # build a custom RC model by adding layers with properties
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(
        RandomReservoirLayer(
            nodes=configuration["nodes"],
            density=configuration["density"],
            activation=configuration["activation"],
            fraction_input=configuration["fraction_input"],
            leakage_rate=configuration["leakage_rate"],
        )
    )
    model.add(ReadoutLayer(output_shape, fraction_out=configuration["fraction_output"]))

    # compile the model
    model.compile(
        optimizer="ridge",
        metrics=configuration["metric"],
        discard_transients=configuration["transients"],
    )

    return model


def fit_evaluate_model(
    data_train: tuple, data_test: tuple, config: dict, graph_analyzer: GraphAnalyzer
):
    """
    Fits and evaluates a classical reservoir computer model
    on the given data set. Returns the model, the score on the test set,
    and the graph properties of the reservoir network.

    Inputs:
    - data_train: tuple containing the training data (x, y)
    - data_test: tuple containing the test data (x, y)
    - config: dictionary containing the configuration of the reservoir
    - graph_analyzer: object that extracts properties of the reservoir graph

    Extract properties of the reservoir using the graph analyzer.
    """

    # unpack data and obtain shapes
    x_train, y_train = data_train
    x_test, y_test = data_test
    input_shape, output_shape = x_train.shape[1:], y_train.shape[1:]

    # build classical RC modeling using a random reservoir graph
    model = build_RC_model(
        input_shape=input_shape,
        output_shape=output_shape,
        configuration=config,
    )

    # train the model
    model.fit(x=x_train, y=y_train)

    # evaluate some metrics on test set
    score = model.evaluate(x=x_test, y=y_test)[0]

    # extract graph-level properties of the reservoir
    graph_props = graph_analyzer.extract_properties(model.reservoir_layer.weights)

    return model, score, graph_props


def prune_evaluate_model(
    model: RC,
    data_train: tuple,
    data_test: tuple,
    pruner: NetworkPruner,
    graph_analyzer: GraphAnalyzer,
):
    """
    Prunes a given model using the pruner object.
    Evaluate the pruned model on the test set.
    Extract properties of the pruned reservoir using the graph analyzer.

    Inputs:
    - model: the model to be pruned (instance of pyreco.models.RC)
    - data_train: tuple containing the training data (x, y)
    - data_test: tuple containing the test data (x, y)
    - pruner: object that prunes the model (instance of pyreco.pruning.NetworkPruner)
    - graph_analyzer: object that extracts properties of the reservoir graph
    """

    # prune the given model
    model_pruned, history = pruner.prune(
        model=model,
        data_train=data_train,
        data_val=data_test,
    )

    print(f"took {history['iteration'][-1]+1} iterations to prune the model")

    # evaluate some metrics on test set
    score = model_pruned.evaluate(*data_test)[0]

    # extract graph-level properties of the reservoir
    graph_props = graph_analyzer.extract_properties(
        model_pruned.reservoir_layer.weights
    )

    return model_pruned, score, graph_props, history
