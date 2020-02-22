import os, sys, threading
from pathlib import Path
import re
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from parse_config import ConfigParser
from utils import util_geo

re_img_index = re.compile("img\d+")


class RenderThread(threading.Thread):
    def __init__(self, img_src, tg_img, geojson_src=None, ras_img=None):
        """
        Args: img_src: Source image path (Raw satellite image)
              tg_img: target image (RGB) output path
              geojson_src:
              resimg_path: raster image (geojson mask) output path
        """
        threading.Thread.__init__(self)
        util = util_geo.GeoLabelUtil()
        self.util = util.GeoImgUtil()
        self.img_src = img_src
        self.tg_img = tg_img
        if geojson_src is not None and ras_img is not None:
            self.geojson_src = geojson_src
            self.ras_img = ras_img
            self.geojson_util = util.GeoJsonUtil()
            self.do_raster = True

    def run(self):
        img = self.util.load_geotiff(self.img_src)
        rimg = self.util.normalize_img(img.ReadAsArray())
        self.util.save(rimg, self.tg_img)
        print("RGB saved to", self.tg_img)
        if self.do_raster:
            f_load = self.geojson_util.load_geojson(self.geojson_src)
            res = self.geojson_util.render(self.util.read_meta(img), f_load.GetLayer())
            self.util.save(res, self.ras_img)
            print("Mask saved to", self.ras_img)


if __name__ == '__main__':
    config = ConfigParser(ConfigParser.from_file("test/configs/geotest.json"))
    logger = config.get_logger('train')
    data_dir = Path(config['data_loader']['args']['data_dir'])
    img_save_dir = Path(config["trainer"]["save_dir"]) / "data" / "RGB"
    data_name = config['data_loader']['args']['data_name']
    img_dir = data_dir / data_name / "RGB-PanSharpen"

    building_geojson_dir = data_dir / data_name / "geojson" / "buildings"
    roads_geojson_dir = data_dir / data_name / "geojson" / "roads"
    raster_save_dir = Path(config["trainer"]["save_dir"]) / "data" / "labels"

    img_save_dir.mkdir(parents=True, exist_ok=True)
    raster_save_dir.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(img_dir):
        img_index = re_img_index.search(f).group(0)
        geojson_fname = "buildings_%s%s.geojson" % (data_name.replace("Train", "").replace("Test", ""), img_index)
        thread = RenderThread(img_dir / f, img_save_dir / f, building_geojson_dir / geojson_fname, raster_save_dir / f)
        thread.run()
