"""fl-thesis: A Flower / PyTorch app with quantization support."""

import logging
from typing import List, Tuple

from flwr.common import Context, Metrics, log, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fl_thesis.custom_strategy import FedAvgWithCost
from fl_thesis.task import get_weights, load_model


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregation function for (weighted) average accuracy.
    """
    accuracies = []
    examples = []

    for num_examples, m in metrics:
        acc_value = float(m.get("accuracy", 0.0))

        accuracies.append(num_examples * acc_value)
        examples.append(num_examples)

    # Hindari pembagian dengan nol jika tidak ada contoh
    if sum(examples) == 0:
        return {"accuracy": 0.0}

    # Hitung rata-rata tertimbang
    weighted_mean = sum(accuracies) / sum(examples)

    return {"accuracy": weighted_mean}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction_evaluate"]
    min_fit_clients = context.run_config["min_fit_clients"]
    min_evaluate_clients = context.run_config["min_evaluate_clients"]
    min_available_clients = context.run_config["min_available_clients"]
    fraction_fit = context.run_config["fraction-fit"]
    quantization = context.run_config.get("quantization", "none")
    quantization_bits = context.run_config.get("quantization-bits", 8)

    # Initialize model parameters and extract shapes for quantization
    model = load_model()
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    log(
        level=logging.INFO,
        msg=f"Server initialized. Quantization: {quantization} ({quantization_bits}-bit)",
    )

    # Define strategy with quantization support
    strategy = FedAvgWithCost(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_fit_clients=min_fit_clients,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        quantization=quantization,
        quantization_bits=quantization_bits,
        num_rounds=num_rounds,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
