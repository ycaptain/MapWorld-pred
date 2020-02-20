from utils import util_geo
from torchvision.datasets import VisionDataset
from PIL import Image


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
    cache = list()
    crs = dict()
    meta = None
    cls_render = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # TODO: read img file
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(GeoDataset, self).__init__(root, transform=transform,
                                         target_transform=target_transform)
        self.data_dir = root
        self.cls_render = util_geo.GeoLabelUtil()
        self.data = dict()
        self.targets = dict()
