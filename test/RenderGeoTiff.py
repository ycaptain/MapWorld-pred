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
    save_dir = data_dir / data_name / 'processed'
    img_save_dir = save_dir / "RGB"

    geojson_dir = data_dir / data_name / "geojson"
    mask_save_dir = save_dir / "labels"
    colors = config['data_loader']['args']["colors"]

    img_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    util_geo.GeoLabelUtil.preprocess(img_dir, geojson_dir, img_save_dir, mask_save_dir, colors)
