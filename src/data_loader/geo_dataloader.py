from base import BaseDataLoader


class GeoDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
