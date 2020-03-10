from .geo_dataset import GeoDataset
import os


class SpaceNetDataset(GeoDataset):

    def __init__(self, **kwargs):
        super(SpaceNetDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, "RGB-PanSharpen")
        self.label_dir = os.path.join(self.root, "geojson")
        # split

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]


