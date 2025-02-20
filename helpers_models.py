from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer


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
