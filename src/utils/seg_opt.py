import cv2
import numpy as np
import json

from PIL import Image
from utils.building_height import random_height


class SegmentOutputUtil:
    def __init__(self, pred, meta, f_augment=None):
        self.pred = pred
        self.meta = meta
        self.f_augment = f_augment

    def set_augment_func(self, f_augment):
        self.f_augment = f_augment

    @staticmethod
    def load_img(path):
        img = Image.open(path)
        # img = Image.eval(img, lambda a: 1 if a >= 128 else 0)
        return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def building_augment(pred):
        kernel = np.ones((5, 5), dtype=np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
        # filter small area
        # Ref: https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 60

        img2 = np.zeros(pred.shape).astype(np.uint8)
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 1
        return img2

    @staticmethod
    def road_augment(pred):
        kernel = np.ones((8, 8), dtype=np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

        # Ref: https://stackoverflow.com/questions/43859750/how-to-connect-broken-lines-in-a-binary-image-using-python-opencv
        kernel = np.ones((1, 15), np.uint8)  # note this is a horizontal kernel
        d_im = cv2.dilate(pred, kernel, iterations=1)
        e_im = cv2.erode(d_im, kernel, iterations=1)

        kernel = np.ones((15, 1), np.uint8)
        d_im = cv2.dilate(e_im, kernel, iterations=1)
        pred = cv2.erode(d_im, kernel, iterations=1)

        return pred

    def get_contours(self, pred):
        if self.f_augment:
            pred = self.f_augment(pred)
        contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt, hie = zip(*filter(lambda c: c[1][3] == -1, zip(contours, *hierarchy)))  # remove inner box
        return cnt, hie

    @staticmethod
    def get_bboxs(cnts):
        bboxs = list()
        for cnt in cnts:
            min_rect = cv2.minAreaRect(cnt)
            min_rect = np.int0(cv2.boxPoints(min_rect))
            bboxs.append(min_rect)
        return bboxs

    @staticmethod
    def show_bbox(rects):
        img = np.zeros((325, 325, 3), np.uint8)
        for rect in rects:
            cv2.drawContours(img, [rect], 0, (0, 255, 0), 2)  # green
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    '''
    3D map JSON format:
    {
        "meta": {
            "x": 0,
            "y": 0
        }
        "building": 
        [
          {
              "coordinates": [
                {
                  "x": 20.355053213979282,
                  "y": -22.450252781908343,
                  "z": 0
                },
                {
                  "x": 20.355053213979282,
                  "y": -17.450252781908343,
                  "z": 0
                },
                {
                  "x": 25.355053213979282,
                  "y": -17.450252781908343,
                  "z": 0
                },
                {
                  "x": 25.355053213979282,
                  "y": -22.450252781908343,
                  "z": 0
                }
              ],
              "properties": {
                "height": 27.832069410709124
              }
          },
          ...
        ],
        "road":
        [
          {
              "coordinates": [
                {
                  "x": 20.355053213979282,
                  "y": -22.450252781908343,
                  "z": 0
                },
                {
                  "x": 20.355053213979282,
                  "y": -17.450252781908343,
                  "z": 0
                },
                {
                  "x": 25.355053213979282,
                  "y": -17.450252781908343,
                  "z": 0
                },
                {
                  "x": 25.355053213979282,
                  "y": -22.450252781908343,
                  "z": 0
                },
              ]
          },
          ...
        ]
    }
    '''

    @staticmethod
    def def_fun_scale(sc, meta):
        # flip y axis
        y = meta["h"]
        sc[1] = y - sc[1]
        sc = sc * 0.1
        return sc

    @staticmethod
    def encoding(items, meta, fun_prop=None, fun_scale=def_fun_scale.__func__):
        res = list()
        for a_item in items:
            targ = dict()
            coords = list()
            for c in a_item:
                coord = dict()
                if fun_scale is not None:
                    c = fun_scale(c, meta)
                coord["x"] = round(float(c[0]), 2)
                coord["y"] = round(float(c[1]), 2)
                coord["z"] = 0
                coords.append(coord)
            targ["coordinates"] = coords
            if fun_prop is not None:
                targ["properties"] = fun_prop(targ, meta)
            res.append(targ)
        return res

    def get_result(self):
        try:
            cnts = self.get_contours(self.pred)
            bboxs = self.get_bboxs(cnts[0])
            building = self.encoding(bboxs, self.meta, fun_prop=random_height)
        except RuntimeError:
            building = list()
        res = dict()
        res["meta"] = self.meta
        res["buildings"] = building
        return json.dumps(res)
