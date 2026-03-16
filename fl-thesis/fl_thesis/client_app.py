"""fl-thesis: A Flower / PyTorch app."""

import logging

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, log
from torchvision.models.resnet import ResNet

from fl_thesis.task import (
    DEVICE,
    get_weights,
    load_data,
    load_model,
    set_weights,
    test,
    train,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net: ResNet, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = DEVICE
        self.net.to(self.device)

    def fit(self, parameters, config):
        log(level=logging.INFO, msg="Memulai proses fit di klien...")

        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    trainloader, valloader = load_data()
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
