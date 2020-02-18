from utils import read_json
from utils import geojson_render
import os


class GeoJsonDataLoader:
    cache = list()
    crs = dict()
    meta = None

    cls_render = None

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cls_render = geojson_render.GeoJsonRender()

    def load_file(self, name):
        raw_json = read_json(os.path.join(self.data_dir, name))
        self.crs = raw_json.get("crs")
        features = raw_json.get("features")
        self.cache = self.cache + features

    def set_metadata(self, meta):
        self.meta = meta

    def render(self, filename, path):
        assert self.meta is not None
        assert "width" in self.meta and "height" in self.meta and "channelsNum" in self.meta
        # Check metadata before we can render
        self.cls_render.create((self.meta["height"], self.meta["width"], self.meta["channelsNum"]))
        for feature in self.cache:
            self.cls_render.draw_item(feature)
        if filename is None and path is None:
            self.cls_render.preview()
        self.cls_render.save(filename, path)
