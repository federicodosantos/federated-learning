"""fl-thesis: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
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
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(load_model())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvgWithCost(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
