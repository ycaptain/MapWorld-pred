import os
import re
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from osgeo import gdal, ogr
from PIL import Image
import threading
from pathlib import Path


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
        def save_rgb(img, filename):
            dist_img = Image.fromarray(img)
            dist_img = dist_img.convert(mode='RGB')
            dist_img.save(str(filename))

        # @staticmethod
        # def save_indexed(img, classes, filename):
        #     dist_img = Image.fromarray(img)
        #     dist_img = dist_img.convert(mode='RGB').convert('P', palette=Image.ADAPTIVE, colors=len(classes))
        #     dist_img.putpalette([item for sublist in classes.values() for item in sublist])
        #     dist_img.save(str(filename))

    class GeoJsonUtil:

        def load_geojson(self, filename):
            # Ref: https://pcjericks.github.io/py-gdalogr-cookbook/layers.html
            # https://www.osgeo.cn/python_gdal_utah_tutorial/ch05.html
            lfile = ogr.Open(str(filename))
            return lfile

        # @staticmethod
        # def render(meta, layer, prev_res=None):
        #     x = meta["RasterXSize"]
        #     y = meta["RasterYSize"]
        #     x_min, x_max, y_min, y_max = layer.GetExtent()
        #     # https://www.programcreek.com/python/example/101827/gdal.RasterizeLayer
        #     driver = gdal.GetDriverByName('MEM')
        #     if prev_res is None:
        #         dst_ds = driver.Create("", x, y, 3, gdal.GDT_Byte)
        #     else:
        #         dst_ds = prev_res
        #     if x_min == 0 and x_max == 0 and y_min == 0 and y_max == 0:
        #         dst_ds.SetGeoTransform(meta["GeoTransform"])
        #     else:
        #         dst_ds.SetGeoTransform((x_min, (x_max - x_min) / x, 0,
        #                                y_max, 0, -(y_max - y_min) / y))
        #     dst_ds.SetProjection(meta["Projection"])
        #     gdal.RasterizeLayer(dst_ds, [1, 2, 3], layer, burn_values=[255])
        #     # img = dst_ds.ReadAsArray()
        #     # img = np.moveaxis(img, 0, -1)
        #     return dst_ds

        @staticmethod
        def render(meta, layer):
            x = meta["RasterXSize"]
            y = meta["RasterYSize"]
            # https://www.programcreek.com/python/example/101827/gdal.RasterizeLayer
            driver = gdal.GetDriverByName('MEM')
            dst_ds = driver.Create("", x, y, 1, gdal.GDT_Byte)
            dst_ds.SetGeoTransform(meta["GeoTransform"])
            dst_ds.SetProjection(meta["Projection"])
            gdal.RasterizeLayer(dst_ds, [1], layer, burn_values=[255])
            img = dst_ds.ReadAsArray()
            return img

    class RenderThread(threading.Thread):
        def __init__(self, img_src, tg_img, cls_gjpath=None, mask_img=None, clses=None):
            """
            Args: img_src: Source image path (Raw satellite image)
                  tg_img: target image (RGB) output path
                  geojson_src:
                  mask_img: geojson mask output path
            """
            threading.Thread.__init__(self)
            util = GeoLabelUtil()
            self.util = util.GeoImgUtil()
            self.img_src = img_src
            self.tg_img = tg_img
            if cls_gjpath is not None and mask_img is not None:
                self.cls_gjpath = cls_gjpath
                self.ras_img = mask_img
                self.geojson_util = util.GeoJsonUtil()
                self.do_mask = True
                self.classes = clses

        def run(self):
            img = self.util.load_geotiff(self.img_src)
            rimg = self.util.normalize_img(img.ReadAsArray())
            self.util.save_rgb(rimg, self.tg_img)
            print("RGB saved to", self.tg_img)
            if self.do_mask:
                meta = self.util.read_meta(img)
                pic_res = Image.new("P", (meta["RasterXSize"], meta["RasterYSize"]))
                # [{"ignore": [0, 0, 0]}, {"building": [255, 0, 0]}, {"road": [0, 0, 255]}]
                pl1 = [item for sublist in self.classes for item in sublist.values()]
                # [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
                # [0, 0, 0, 255, 0, 0, 0, 0, 255]
                pic_res.putpalette([item for sublist in pl1 for item in sublist])
                for k, v in self.cls_gjpath.items():
                    # cls_gj: {"building": "****_img1.geojson", "road": "****_img1.geojson"}
                    f_load = self.geojson_util.load_geojson(v)
                    if f_load is not None:
                        rend = self.geojson_util.render(meta, f_load.GetLayer())
                        if rend is not None:
                            rend = Image.fromarray(rend).convert('L')
                            pic_res.paste([i for i, d in enumerate(self.classes) if k in d.keys()][0], rend)
                            pic_res.save(self.ras_img, optimize=1)
                            print("Mask saved to", self.ras_img)

    @staticmethod
    def preprocess(img_dir, geojson_dir, rgb_tg, mask_tg, classes, num_workers=4):
        pool = ThreadPoolExecutor(max_workers=num_workers)
        re_img_index = re.compile("img\d+")
        re_pat = re.compile("(.*?)img\d+")
        building_pat = re_pat.search(os.listdir(Path(geojson_dir) / "buildings")[0]).group(1)
        # road_pat = re_pat.search(os.listdir(geojson_dir / "roads")[0]).group(1)

        for f in os.listdir(img_dir):
            img_index = re_img_index.search(f).group(0)
            geojsons = {"building": Path(geojson_dir) / "buildings" / (building_pat + img_index + ".geojson")}

            # geojsons = {"road": Path(geojson_dir) / "roads" / (road_pat + img_index + ".geojson")}
            # geojsons = {"building": Path(geojson_dir) / "buildings" / (building_pat + img_index + ".geojson"),
            #             "road": Path(geojson_dir) / "roads" / (road_pat + img_index + ".geojson")}

            def pool_wrapper(p1, p2, p3, p4, p5):
                thread = GeoLabelUtil.RenderThread(p1, p2, p3, p4, p5)
                thread.run()
                # thread.start()

            pool.submit(pool_wrapper, Path(img_dir) / f,
                        Path(rgb_tg) / (img_index + ".png"),
                        geojsons,
                        Path(mask_tg) / (img_index + ".png"), classes)
        pool.shutdown(wait=True)
