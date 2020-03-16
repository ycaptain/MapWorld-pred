import numpy as np
from torchvision.datasets import VisionDataset
import os
from pathlib import Path


class GeoDataset(VisionDataset):
    """GeoDataset.
        Args:
            root (string): Root directory of dataset.
            train (bool, optional): If True, creates dataset from ``training``,
                otherwise from ``test``.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
    files = []

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # TODO: read img file
        image_id, image, label = self._load_data(index)

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(label)

        return image_id, image.astype(np.float32), label.astype(np.int64)

    @property
    def processed_folder(self):
        return Path(os.path.join(self.root, 'processed'))

    def __init__(self, root, colors, train=True, transform=None, target_transform=None):
        super(GeoDataset, self).__init__(root, transform=transform,
                                         target_transform=target_transform)
        self.train = train  # training set or test set
        self.colors = colors
        self.files = []
        self._set_files()

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
