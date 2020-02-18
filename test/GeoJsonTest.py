import unittest
import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from data_loader import geojson
from parse_config import ConfigParser


class GeoJsonTestCase(unittest.TestCase):
    config = None
    loader = None
    meta = {
        "height": 650,
        "width": 650,
        "channelsNum": 3,
        "origin": (2.186411399970000, 49.051215900000003)
    }

    def __init__(self, *args, **kwargs):
        super(GeoJsonTestCase, self).__init__(*args, **kwargs)
        config = ConfigParser(ConfigParser.from_file("test/configs/geojson.json"))
        logger = config.get_logger('train')
        self.config = config

    def test_loader(self):
        loader = geojson.GeoJsonDataLoader(self.config['data_loader']['args']['data_dir'])
        self.assertEqual(loader.data_dir, "data/FYPData/spacenet/AOI_3_Paris_Train/geojson/buildings")
        self.loader = loader
        t = loader.load_file("buildings_AOI_3_Paris_img19.geojson")
        print(loader.cache)
        loader.set_metadata(self.meta)
        #loader.render("test.png", root_dir)
        loader.render(None, None)  # Preview


if __name__ == '__main__':
    unittest.main()
