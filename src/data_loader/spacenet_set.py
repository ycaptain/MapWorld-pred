import os
from .geo_dataset import GeoDataset
from utils import util_geo
from PIL import Image


class SpaceNetDataset(GeoDataset):

    def __init__(self, data_type, **kwargs):
        self.data_type = data_type
        super(SpaceNetDataset, self).__init__(**kwargs)

    @property
    def train_file(self):
        return os.path.join(self.processed_folder, "training.txt")

    def _check_exists(self):
        return os.path.exists(self.train_file)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, "RGB-PanSharpen")
        self.label_dir = os.path.join(self.root, "geojson")
        if not self._check_exists():
            self.process()
        file_list = tuple(open(self.train_file, "r"))
        file_list = [id_.rstrip() for id_ in file_list]  # remove \n

        # update dir
        self.image_dir = self.processed_folder / "RGB"
        self.label_dir = self.processed_folder / "labels"
        self.files = file_list

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id)
        label_path = os.path.join(self.label_dir, image_id)
        image = Image.open(image_path)
        label = Image.open(label_path)
        return os.path.basename(self.root) + "_" + os.path.splitext(image_id)[0], image, label

    def process(self):
        img_save_dir = self.processed_folder / "RGB"
        img_save_dir.mkdir(parents=True, exist_ok=True)
        mask_save_dir = self.processed_folder / "labels"
        mask_save_dir.mkdir(parents=True, exist_ok=True)
        util_geo.GeoLabelUtil.preprocess(self.data_type, self.image_dir, self.label_dir, img_save_dir, mask_save_dir)
        # split to list
        flists = os.listdir(img_save_dir)
        with open(self.train_file, 'w') as f:
            f.writelines("%s\n" % img for img in flists)
