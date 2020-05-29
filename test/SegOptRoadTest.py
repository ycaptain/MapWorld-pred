import unittest
import os
import sys
import cv2
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
    def setUpClass(self):
        pred = SegmentOutputUtil.load_img("tmp/results/a00d13ba_Road-Deeplab_0.png")
        w, h = pred.shape

        self.meta = {
            "w": w,
            "h": h
        }

        self.util = SegmentOutputUtil(pred, self.meta, "Road")
        self.pred = self.util.f_augment(pred)

    def test_road_postprocess(self):
        skel = self.pred
        for i in range(2):
            skel = self.util.get_skeleton(skel)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for i in range(4):
                skel = cv2.dilate(skel, kernel, iterations=5)
                skel = SegmentOutputUtil.connect_line(skel, 10, 1)

        mid_line = self.util.get_skeleton(skel)
        img = np.zeros(skel.shape).astype(np.uint8)
        img[skel == 255] = [50]
        img[mid_line == 255] = [255]

        alpha = Image.fromarray(skel)
        img = Image.merge('LA', [Image.fromarray(img), alpha])
        img.show()


if __name__ == '__main__':
    unittest.main()
