import numpy as np
import torch.utils.data as data
import os
from pathlib import Path


class GeoDataset(data.Dataset):
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

    def __init__(self, root, classes, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = classes
        self.files = []
        self._set_files()

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
            tuple: (image_id, image, target)
        """
        image_id, image, label = self._load_data(index)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        image = np.asarray(image)
        label = np.asarray(label, dtype=np.uint8)
        return image_id, np.moveaxis(image, -1, 0).astype(np.float32), label.astype(np.int64)

    @property
    def processed_folder(self):
        return Path(os.path.join(self.root, 'processed'))

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
