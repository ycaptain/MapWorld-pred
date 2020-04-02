from .spacenet_loader import SpaceNetDataLoader


def get_dataloader(name):
    return {
        "SpaceNet": SpaceNetDataLoader,
    }[name]
