import unittest
from pathlib import Path
import os, sys

import numpy as np

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
    meta = None

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
        img_util = self.util.GeoImgUtil()
        timg = img_util.load_geotiff(imgdir / "RGB-PanSharpen_AOI_3_Paris_img100.tif")
        img = img_util.normalize_img(timg.ReadAsArray())
        print("Metadata: ", img_util.read_meta(timg))
        img_util.preview(img)

    def test_geojson(self):
        imgdir = self.datadir / self.data_name / "RGB-PanSharpen"
        img_util = self.util.GeoImgUtil()
        timg = img_util.load_geotiff(imgdir / "RGB-PanSharpen_AOI_3_Paris_img100.tif")
        # save_dir = Path(self.config["trainer"]["save_dir"]) / "data" / "labels"
        # save_dir.mkdir(parents=True, exist_ok=True)

        building_jsondir = self.datadir / self.data_name / "geojson" / "buildings"
        road_jsondir = self.datadir / self.data_name / "geojson" / "roads"
        geojson_util = self.util.GeoJsonUtil()
        rf = geojson_util.load_geojson(road_jsondir / "SN3_roads_train_AOI_3_Paris_geojson_roads_img100.geojson")
        f = geojson_util.load_geojson(building_jsondir / "buildings_AOI_3_Paris_img100.geojson")
        res_img = geojson_util.render(img_util.read_meta(timg), f.GetLayer())
        img_util.preview(res_img)


if __name__ == '__main__':
    unittest.main()
