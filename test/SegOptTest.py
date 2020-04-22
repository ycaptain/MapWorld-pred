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

    @classmethod
    def setUpClass(cls):
        # preds = []
        # pred = cls.load_img("data/FYPData/1.png")
        pred = SegmentOutputUtil.load_img("data/FYPData/2.png")
        # preds.append(cls.load_img("data/FYPData/2.png"))
        # preds.append(cls.load_img("data/FYPData/3.png"))
        # preds.append(cls.load_img("data/FYPData/4.png"))
        # preds.append(cls.load_img("data/FYPData/5.png"))
        cls.util = SegmentOutputUtil(pred)
        cls.pred = pred

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


if __name__ == '__main__':
    unittest.main()
