import numpy as np
from osgeo import gdal, ogr
from PIL import Image
import threading


class GeoLabelUtil:

    class GeoImgUtil:
        img = None
        meta = None

        def load_geotiff(self, filename):
            img = gdal.Open(str(filename))
            self.meta = self.read_meta(img)
            self.img = img
            return self.img

        def get_meta(self):
            return self.meta

        @staticmethod
        def read_meta(img):
            meta = img.GetMetadata()
            meta["RasterXSize"] = img.RasterXSize
            meta["RasterYSize"] = img.RasterYSize
            meta["RasterCount"] = img.RasterCount
            meta["GeoTransform"] = img.GetGeoTransform()
            meta["Projection"] = img.GetProjection()
            return meta

        @staticmethod
        def normalize_img(img, xtimes=2.5):
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

            return img

        @staticmethod
        def preview(img):
            Image.fromarray(img).show()

        @staticmethod
        def save(img, filename):
            dist_img = Image.fromarray(img)
            dist_img = dist_img.convert(mode='RGB')
            dist_img.save(str(filename))

    class GeoJsonUtil:

        def load_geojson(self, filename):
            # Ref: https://pcjericks.github.io/py-gdalogr-cookbook/layers.html
            # https://www.osgeo.cn/python_gdal_utah_tutorial/ch05.html
            lfile = ogr.Open(str(filename))
            return lfile

        @staticmethod
        def render(meta, layer, color=None):
            if color is None:
                color = [255, 255, 255]
            x = meta["RasterXSize"]
            y = meta["RasterYSize"]
            # https://www.programcreek.com/python/example/101827/gdal.RasterizeLayer
            driver = gdal.GetDriverByName('MEM')
            dst_ds = driver.Create("", x, y, 3, gdal.GDT_Byte)
            dst_ds.SetGeoTransform(meta["GeoTransform"])
            dst_ds.SetProjection(meta["Projection"])
            gdal.RasterizeLayer(dst_ds, [1, 2, 3], layer, burn_values=color)
            img = dst_ds.ReadAsArray()
            img = np.moveaxis(img, 0, -1)
            return img

    class RenderThread(threading.Thread):
        def __init__(self, img_src, tg_img, geojson_src=None, ras_img=None, cols=None):
            """
            Args: img_src: Source image path (Raw satellite image)
                  tg_img: target image (RGB) output path
                  geojson_src:
                  resimg_path: raster image (geojson mask) output path
            """
            threading.Thread.__init__(self)
            util = GeoLabelUtil()
            self.util = util.GeoImgUtil()
            self.img_src = img_src
            self.tg_img = tg_img
            if geojson_src is not None and ras_img is not None:
                self.geojson_src = geojson_src
                self.ras_img = ras_img
                self.geojson_util = util.GeoJsonUtil()
                self.do_raster = True
                self.colors = cols

        def run(self):
            img = self.util.load_geotiff(self.img_src)
            rimg = self.util.normalize_img(img.ReadAsArray())
            self.util.save(rimg, self.tg_img)
            print("RGB saved to", self.tg_img)
            if self.do_raster:
                f_load = self.geojson_util.load_geojson(self.geojson_src)
                res = self.geojson_util.render(self.util.read_meta(img), f_load.GetLayer(), self.colors["building"])
                self.util.save(res, self.ras_img)
                print("Mask saved to", self.ras_img)
