import os, sys
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

if __name__ == '__main__':
    config = ConfigParser(ConfigParser.from_file("test/configs/geotest.json"))
    logger = config.get_logger('train')
    data_dir = Path(config['data_loader']['args']['data_dir'])
    data_name = config['data_loader']['args']['data_name']
    img_dir = data_dir / data_name / "RGB-PanSharpen"
    save_dir = data_dir / data_name / 'SpaceNetDataset' / 'processed'
    img_save_dir = save_dir / "RGB"

    building_geojson_dir = data_dir / data_name / "geojson" / "buildings"
    roads_geojson_dir = data_dir / data_name / "geojson" / "roads"
    raster_save_dir = save_dir / "labels"
    colors = config['data_loader']['args']["colors"]

    img_save_dir.mkdir(parents=True, exist_ok=True)
    raster_save_dir.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(img_dir):
        img_index = re_img_index.search(f).group(0)
        geojson_fname = "buildings_%s%s.geojson" % (data_name.replace("Train", "").replace("Test", ""), img_index)
        thread = util_geo.GeoLabelUtil.RenderThread(img_dir / f, img_save_dir / f, building_geojson_dir / geojson_fname, raster_save_dir / f, colors)
        thread.run()
