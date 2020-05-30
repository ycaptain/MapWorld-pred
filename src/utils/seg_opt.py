import cv2
import numpy as np
import json
import os

from PIL import Image
from utils.building_height import random_height
from utils.util import read_json


class SegmentOutputUtil:
    def __init__(self, pred, meta, mtype="Building"):
        self.pred = pred
        self.meta = meta
        self.f_augment = None
        self.type = None
        self.set_type(mtype)

    def set_type(self, mtype):
        if "Building" in mtype:
            self.type = "Building"
            self.f_augment = self.building_augment
        elif "Road" in mtype:
            self.type = "Road"
            self.f_augment = self.road_augment

    @staticmethod
    def load_img(path):
        img = Image.open(path)
        # img = Image.eval(img, lambda a: 1 if a >= 128 else 0)
        return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def filter_size(pred, min_size, set_value=1):
        # filter small area
        # Ref: https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv

        # min_size minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        img2 = np.zeros(pred.shape).astype(np.uint8)
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = set_value
        return img2

    @staticmethod
    def building_augment(pred):
        kernel = np.ones((4, 4), dtype=np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
        return SegmentOutputUtil.filter_size(pred, 60)

    @staticmethod
    def connect_line(pred, n, i, n2=1):
        # Ref: https://stackoverflow.com/questions/43859750/how-to-connect-broken-lines-in-a-binary-image-using-python-opencv
        kernel = np.ones((n2, n), np.uint8)  # note this is a horizontal kernel
        d_im = cv2.dilate(pred, kernel, iterations=i)
        e_im = cv2.erode(d_im, kernel, iterations=i)

        kernel = np.ones((n, n2), np.uint8)
        d_im = cv2.dilate(e_im, kernel, iterations=i)
        pred = cv2.erode(d_im, kernel, iterations=i)
        ret, pred = cv2.threshold(pred, 100, 255, cv2.THRESH_BINARY)

        return pred

    @staticmethod
    def road_augment(pred):
        kernel = np.ones((4, 4), dtype=np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
        pred = SegmentOutputUtil.connect_line(pred, 4, 5)
        return pred

    @staticmethod
    def get_contours(pred):
        contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt, hie = zip(*filter(lambda c: c[1][3] == -1, zip(contours, *hierarchy)))  # remove inner box
        return cnt, hie

    @staticmethod
    def get_skeleton(pred):
        # Ref: https://stackoverflow.com/questions/33095476/is-there-any-build-in-function-can-do-skeletonization-in-opencv
        size = np.size(pred)
        skel = np.zeros(pred.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(pred, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(pred, temp)
            skel = cv2.bitwise_or(skel, temp)
            pred = eroded.copy()

            zeros = size - cv2.countNonZero(pred)
            if zeros == size:
                done = True

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        skel = cv2.dilate(skel, kernel, iterations=1)

        skel = SegmentOutputUtil.connect_line(skel, 4, 1)

        skel = SegmentOutputUtil.filter_size(skel, 60, set_value=255)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # for i in range(0, 10):
        #     skel = cv2.dilate(skel, kernel, iterations=1)
        #     skel = SegmentOutputUtil.connect_line(skel, 5, 1)

        return skel

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

    def get_result(self, target):
        if self.f_augment:
            self.pred = self.f_augment(self.pred)
        # load previous result
        if os.path.isfile(target+".json"):
            res = read_json(target+".json")
            if not "meta" in res:
                print("Warning: no meta in exist json file.")
            elif self.meta != res["meta"]:
                print("Warning: meta for the current image is not same as previous result, may result in wrong align.")
        else:
            res = dict()
        if self.type == "Building":
            try:
                cnts = self.get_contours(self.pred)
                bboxs = self.get_bboxs(cnts[0])
                building = self.encoding(bboxs, self.meta, fun_prop=random_height)
            except (RuntimeError, TypeError):
                building = list()
            res["meta"] = self.meta
            res["buildings"] = building
            with open(target+".json", 'w') as f:
                f.write(json.dumps(res))

        elif self.type == "Road":
            res["meta"] = self.meta
            try:
                skel = self.pred
                for i in range(2):
                    skel = self.get_skeleton(skel)

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    for i in range(4):
                        skel = cv2.dilate(skel, kernel, iterations=5)
                        skel = SegmentOutputUtil.connect_line(skel, 10, 1)

                mid_line = self.get_skeleton(skel)
                img = np.zeros(skel.shape).astype(np.uint8)
                img[skel == 255] = [50]
                img[mid_line == 255] = [255]

                alpha = Image.fromarray(skel)
                img = Image.merge('LA', [Image.fromarray(img), alpha])
                img_path = target+"_road.png"
                img.save(img_path)
                res["roadImg"] = img_path

            except (RuntimeError, TypeError):
                res["roadImg"] = ""

            with open(target+".json", 'w') as f:
                f.write(json.dumps(res))

        return target+".json"