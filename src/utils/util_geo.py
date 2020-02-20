import cv2
import numpy as np
import os
from . import read_json
from osgeo import gdal, osr, ogr


class GeoLabelUtil:

    class GeoImgUtil:
        img = None

        def __init__(self, data_dir):
            self.data_dir = data_dir

        def load_geotiff(self, filename):
            img = gdal.Open(os.path.join(self.data_dir, filename)).ReadAsArray()
            self.img = self.normalize_img(img)
            return self.img

        def normalize_img(self, img, xtimes=2.5):
            """
            Image data before normalize, 3 channels shaped as (650,650,3):
            [[365 367 359...(x650)], [368 361 367...]...(x650)]
            [[391 394 387...], [395 389 394...]...]
            [[308 310 306...], [310 306 309...]...]
            1. Reshape matrix:
                change the data shape to (3, 650, 650) then we can get following:
                [[365 391 308], [367 394 310], [359 387 306]...x650]
                [[368 395 310], [361 389 306], [367 394 309]...x650]
                ...x650, which resembles to a image with 650x650 pixels.
            2. Get Max-Min:
                Calculate mean of nonzero values, then lower and upper bond is +/- X times standard deviation.
            3. Scale the image to Max-Min
            """
            img = np.moveaxis(img, 0, -1)  # reshape
            # Ref: https://stackoverflow.com/questions/38542548/numpy-mean-of-nonzero-values
            # means = np.true_divide(img.sum(axis=(0, 1)), (img != 0).sum(axis=(0, 1)))
            means = np.nanmean(np.where(np.isclose(img, 0), np.nan, img), axis=(0, 1))
            stds = np.nanstd(np.where(np.isclose(img, 0), np.nan, img), axis=(0, 1))
            mins = means - xtimes * stds
            maxs = means + xtimes * stds

            img = (img - mins) / (maxs - mins) * 255
            img = np.uint8(np.clip(img, 0, 255, img))

            self.img = img
            return img

        @staticmethod
        def preview(img):
            cv2.imshow('Preview', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        @staticmethod
        def save(img, filename, path):
            cv2.imwrite(os.path.join(path, filename), img)

    class GeoJsonUtil:
        crs = ""
        cache = list()
        meta = None
        img = None

        def __init__(self, data_dir):
            self.data_dir = data_dir

        def load_geojson(self, name):
            raw_json = read_json(os.path.join(self.data_dir, name))
            self.crs = raw_json.get("crs")
            features = raw_json.get("features")
            self.cache = self.cache + features
            return raw_json

        def set_metadata(self, meta):
            self.meta = meta

        def render(self, filename, path):
            assert self.meta is not None
            assert "width" in self.meta and "height" in self.meta and "channelsNum" in self.meta
            # Check metadata before we can render
            self.create((self.meta["height"], self.meta["width"], self.meta["channelsNum"]))
            for feature in self.cache:
                self.draw_item(feature)
            if filename is None or path is None:
                GeoLabelUtil.GeoImgUtil.preview(self.img)
            else:
                self.save(self.img, filename, path)

        def create(self, prop):
            self.img = np.zeros(prop, dtype="uint8")

        def draw_item(self, item):
            pass

