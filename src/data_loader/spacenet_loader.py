from torchvision import transforms
from base import BaseDataLoader
from .spacenet_set import SpaceNetDataset
import os.path


class SpaceNetDataLoader(BaseDataLoader):

    def __init__(self, data_name, data_dir, batch_size, colors=None, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if colors is None:
            colors = {
                "building": [255, 0, 0],
                "road": [0, 0, 255],
                "ignore": [0, 0, 0]
            }
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_name = data_name
        self.data_dir = data_dir
        self.colors = colors
        self.dataset = SpaceNetDataset(root=os.path.join(self.data_dir, self.data_name), colors=colors, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
