import os, sys, threading
from pathlib import Path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from parse_config import ConfigParser
from utils import util_geo


class RenderThread(threading.Thread):
    def __init__(self, imgdir, name, path):
        threading.Thread.__init__(self)
        util = util_geo.GeoLabelUtil()
        self.util = util.GeoImgUtil(imgdir)
        self.name = name
        self.path = path

    def run(self):
        img = self.util.load_geotiff(self.name)
        self.util.save(img, self.name, self.path)
        print(self.name, "saved to ", self.path)


if __name__ == '__main__':
    config = ConfigParser(ConfigParser.from_file("test/configs/geotest.json"))
    logger = config.get_logger('train')
    data_dir = Path(config['data_loader']['args']['data_dir'])
    save_dir = Path(config["trainer"]["save_dir"]) / "data"
    data_name = config['data_loader']['args']['data_name']

    imgdir = data_dir / data_name / "RGB-PanSharpen"
    save_dir.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(imgdir):
        thread = RenderThread(imgdir, f, save_dir)
        thread.run()
