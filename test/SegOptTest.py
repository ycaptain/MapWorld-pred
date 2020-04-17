import unittest
import os
import sys
import numpy as np
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from utils.seg_opt import SegmentOutputUtil


class SegOptTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        preds = []
        preds.append(cls.load_img("data/FYPData/1.png"))
        preds.append(cls.load_img("data/FYPData/2.png"))
        preds.append(cls.load_img("data/FYPData/3.png"))
        preds.append(cls.load_img("data/FYPData/4.png"))
        preds.append(cls.load_img("data/FYPData/5.png"))
        cls.util = SegmentOutputUtil(None, preds)
        cls.preds = preds

    @staticmethod
    def load_img(path):
        img = Image.open(path)
        # img = Image.eval(img, lambda a: 1 if a == 255 else 0)
        return np.asarray(img, dtype=np.uint8)

    def test_bbox(self):
        res = self.util.get_bounding_box(self.preds)
        for id, (img, bbox) in enumerate(zip(self.preds, res)):
            self.util.show_bbox(img, bbox[0])


if __name__ == '__main__':
    unittest.main()
