import numpy as np
from osgeo import gdal, ogr
from PIL import Image


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

        color_map = dict()

        def load_geojson(self, filename):
            # Ref: https://pcjericks.github.io/py-gdalogr-cookbook/layers.html
            # https://www.osgeo.cn/python_gdal_utah_tutorial/ch05.html
            lfile = ogr.Open(str(filename))
            return lfile

        @staticmethod
        def render(meta, layer):
            x = meta["RasterXSize"]
            y = meta["RasterYSize"]
            # https://www.programcreek.com/python/example/101827/gdal.RasterizeLayer
            driver = gdal.GetDriverByName('MEM')
            dst_ds = driver.Create("", x, y, 1, gdal.GDT_UInt16)
            dst_ds.SetGeoTransform(meta["GeoTransform"])
            dst_ds.SetProjection(meta["Projection"])
            gdal.RasterizeLayer(dst_ds, [1], layer, None)
            img = dst_ds.ReadAsArray()
            return img
