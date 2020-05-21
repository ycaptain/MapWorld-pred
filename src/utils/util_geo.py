import os
import re
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from osgeo import gdal, ogr
from PIL import Image
import threading
from pathlib import Path
import cv2
from urllib import request
from urllib import parse


class GeoLabelUtil(object):
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

        @staticmethod
        def download_from_gmap(path, center_long, center_lat, zoom=19, size_w=600, size_h=600, type="roadmap", format="png"):
            # https://maps.googleapis.com/maps/api/staticmap?center=36.1695402,-115.3066401&zoom=19&size=600x625&maptype=roadmap&format=png&style=feature:all|element:labels|visibility:off&key=AIzaSyA0RZ-MAiBwIrZM8NK8ORqC_gRrFjFs5Y0
            MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"
            API_KEY = "AIzaSyA0RZ-MAiBwIrZM8NK8ORqC_gRrFjFs5Y0"
            params = parse.urlencode({'center': "%f,%f" % (center_long, center_lat),
                                      'zoom': zoom,
                                      'size': "%dx%d" % (size_w, size_h+25),
                                      'type': type,
                                      'format': format,
                                      'style': 'feature:all|element:labels|visibility:off',
                                      'key': API_KEY})
            req = request.Request(MAP_URL+"?%s" % params, method="GET")
            with request.urlopen(req) as response, open(path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            # crop google mark
            im = Image.open(path)
            width, height = im.size
            im1 = im.crop((0, 0, width, height-25))
            im1.convert('RGB').save(path, ptimize=1)
            print("Downloaded to %s" % path)

    class BuildingRenderThread(threading.Thread):
        def __init__(self, img_src, tg_img, gjpath=None, mask_img=None):
            """
            Args: img_src: Source image path (Raw satellite image)
                  tg_img: target image (RGB) output path
                  gjpath: location to the geojson file
                  mask_img: geojson mask output path
            """
            threading.Thread.__init__(self)
            util = GeoLabelUtil()
            self.util = util.GeoImgUtil()
            self.img_src = img_src
            self.tg_img = tg_img
            if gjpath is not None and mask_img is not None:
                self.gjpath = gjpath
                self.ras_img = mask_img
                self.do_mask = True

        def render(self, meta, gjpath):
            lfile = ogr.Open(str(gjpath))
            if lfile is None:
                return None
            layer = lfile.GetLayer()
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

        def run(self):
            img = self.util.load_geotiff(self.img_src)
            rimg = self.util.normalize_img(img.ReadAsArray())
            self.util.save_rgb(rimg, self.tg_img)
            print("RGB saved to", self.tg_img)
            if self.do_mask:
                meta = self.util.read_meta(img)
                pic_res = Image.new("P", (meta["RasterXSize"], meta["RasterYSize"]))
                pic_res.putpalette([0, 0, 0, 255, 255, 255])
                rend = self.render(meta, self.gjpath)
                if rend is not None:
                    rend = Image.fromarray(rend).convert('L')
                else:
                    rend = Image.new("L", (meta["RasterXSize"], meta["RasterYSize"]))
                pic_res.paste(1, rend)
                pic_res.save(self.ras_img, optimize=1)
                print("Mask saved to", self.ras_img)

    class RoadRenderThread(BuildingRenderThread):
        def render(self, meta, gjpath):
            img = super().render(meta, gjpath)
            if img is None:
                return None
            # Ref: https://xbuba.com/questions/46895772
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            dilate = cv2.dilate(img, kernel, iterations=1)
            return dilate

    @staticmethod
    def preprocess(data_type, img_dir, geojson_dir, rgb_tg, mask_tg, num_workers=8):
        pool = ThreadPoolExecutor(max_workers=num_workers)
        re_img_index = re.compile("img\d+")
        re_pat = re.compile("(.*?)img\d+")
        # building_pat = re_pat.search(os.listdir(Path(geojson_dir) / "buildings")[0]).group(1)
        pat = re_pat.search(os.listdir(Path(geojson_dir))[0]).group(1)

        for f in os.listdir(img_dir):
            img_index = re_img_index.search(f).group(0)
            geojson = Path(geojson_dir) / (pat + img_index + ".geojson")

            def pool_wrapper(p1, p2, p3, p4):
                if data_type == "building":
                    thread = GeoLabelUtil.BuildingRenderThread(p1, p2, p3, p4)
                elif data_type == "road":
                    thread = GeoLabelUtil.RoadRenderThread(p1, p2, p3, p4)
                else:
                    raise NotImplementedError("Not Implemented Data Type: " + data_type)
                thread.run()
                # thread.start()

            pool.submit(pool_wrapper, Path(img_dir) / f,
                        Path(rgb_tg) / (img_index + ".png"),
                        geojson,
                        Path(mask_tg) / (img_index + ".png"))
        pool.shutdown(wait=True)
