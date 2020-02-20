import unittest
from pathlib import Path
import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from parse_config import ConfigParser
from utils import util_geo


class GeoTestCase(unittest.TestCase):
    config = None
    loader = None
    meta = {
        "height": 650,
        "width": 650,
        "channelsNum": 3,
        "origin": (2.186411399970000, 49.051215900000003)
    }

    @classmethod
    def setUpClass(cls):
        config = ConfigParser(ConfigParser.from_file("test/configs/geotest.json"))
        logger = config.get_logger('train')
        cls.config = config
        cls.datadir = Path(config['data_loader']['args']['data_dir'])
        cls.data_name = config['data_loader']['args']['data_name']
        cls.util = util_geo.GeoLabelUtil()

    def test_img(self):
        imgdir = self.datadir / self.data_name / "RGB-PanSharpen"
        img_util = self.util.GeoImgUtil(imgdir)
        timg = img_util.load_geotiff("RGB-PanSharpen_AOI_3_Paris_img19.tif")
        img_util.preview(timg)

    def test_geojson(self):
        building_jsondir = self.datadir / self.data_name / "geojson" / "buildings"
        json_util = self.util.GeoJsonUtil(building_jsondir)
        f = json_util.load_geojson("buildings_AOI_3_Paris_img19.geojson")
        # print(loader.cache)
        json_util.set_metadata(self.meta)
        # loader.render("test.png", root_dir)
        json_util.render(None, None)  # Preview


if __name__ == '__main__':
    unittest.main()
