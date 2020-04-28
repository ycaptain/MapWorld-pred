import unittest
import os
import sys
import json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from utils.seg_opt import SegmentOutputUtil
from utils.building_height import random_height


class SegOptTest(unittest.TestCase):

    def test_load_building(self):
        pred = SegmentOutputUtil.load_img("tmp/results/0467da24_Building-Deeplab_1.png")
        w, h = pred.shape

        self.meta = {
            "w": w,
            "h": h
        }

        self.util = SegmentOutputUtil(pred, self.meta)
        self.pred = pred

    def test_bbox(self):
        cnts = self.util.get_contours(self.pred)
        bboxs = self.util.get_bboxs(cnts[0])
        self.util.show_bbox(bboxs)

    def test_encoding(self):
        cnts = self.util.get_contours(self.pred)
        bboxs = self.util.get_bboxs(cnts[0])

        buildings = self.util.encoding(bboxs, self.util.meta, fun_prop=random_height)
        res = json.dumps({
            "meta": self.meta,
            "buildings": buildings,
            "road": [
                {
                    "coordinates": [
                        {
                            "x": 0.0,
                            "y": 5.2,
                            "z": 0
                        },
                        {
                            "x": 32.5,
                            "y": 5.9,
                            "z": 0
                        }
                    ]
                },
                {
                    "coordinates": [
                        {
                            "x": 0.0,
                            "y": 14.3,
                            "z": 0
                        },
                        {
                            "x": 32.5,
                            "y": 15.45,
                            "z": 0
                        }
                    ]
                },
                {
                    "coordinates": [
                        {
                            "x": 0.0,
                            "y": 23.45,
                            "z": 0
                        },
                        {
                            "x": 32.5,
                            "y": 23.85,
                            "z": 0
                        }
                    ]
                }
            ]
        })
        print(res)
        with open('test/test_res.json', 'w') as f:
            f.write(res)

    def test_load_road(self):
        pred = SegmentOutputUtil.load_img("tmp/results/0467da24_Road-Deeplab_1.png")
        w, h = pred.shape
        self.meta = {
            "w": w,
            "h": h
        }

        self.util = SegmentOutputUtil(pred, self.meta)
        self.pred = pred


if __name__ == '__main__':
    unittest.main()
