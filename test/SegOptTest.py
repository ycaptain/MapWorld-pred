import unittest
import os
import sys
import json
import numpy as np
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from utils.seg_opt import SegmentOutputUtil
from utils.building_height import random_height


class SegOptTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # preds = []
        pred = cls.load_img("data/FYPData/1.png")
        # preds.append(cls.load_img("data/FYPData/2.png"))
        # preds.append(cls.load_img("data/FYPData/3.png"))
        # preds.append(cls.load_img("data/FYPData/4.png"))
        # preds.append(cls.load_img("data/FYPData/5.png"))
        cls.util = SegmentOutputUtil(None, pred)
        cls.pred = pred

    @staticmethod
    def load_img(path):
        img = Image.open(path)
        # img = Image.eval(img, lambda a: 1 if a == 255 else 0)
        return np.asarray(img, dtype=np.uint8)

    def test_bbox(self):
        cnts = self.util.get_contours(self.pred)
        bboxs = self.util.get_bboxs(cnts[0])
        self.util.show_bbox(self.pred, bboxs)

    def test_encoding(self):
        cnts = self.util.get_contours(self.pred)
        bboxs = self.util.get_bboxs(cnts[0])

        def scalar(sc, meta):
            return sc * 0.1

        buildings = self.util.encoding(bboxs, fun_prop=random_height, fun_scale=scalar)
        res = json.dumps({
            "building": buildings
        })
        print(res)
        with open('test/test_res.json', 'w') as f:
            f.write(res)


if __name__ == '__main__':
    unittest.main()
