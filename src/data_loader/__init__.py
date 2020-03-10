from .mnist_loaders import MnistDataLoader
from .spacenet_loader import SpaceNetDataLoader


def get_dataloader(name):
    return {
        "Mnist": MnistDataLoader,
        "SpaceNet": SpaceNetDataLoader,
    }[name]
