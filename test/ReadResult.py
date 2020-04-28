import unittest
import cv2
import numpy as np

from utils.util import read_json


class ReadResultTest(unittest.TestCase):

    def test_draw(self):
        j = read_json("test_res.json")
        img = np.zeros((j["meta"]["h"], j["meta"]["w"], 3), np.uint8)
        for building in j["buildings"]:
            rect = list()
            for coor in building["coordinates"]:
                rect.append([coor["x"], coor["y"]])
            narray = np.array(rect)
            narray *= 10
            narray = narray.astype(np.int)
            cv2.drawContours(img, [narray], 0, (0, 255, 0), 2)  # green
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()