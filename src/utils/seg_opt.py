import cv2
import numpy as np
import json


class SegmentOutputUtil:
    def __init__(self, raw_img, preds):
        self.raw_img = raw_img
        self.preds = preds

    def pixelize(self, preds):
        return preds

    @staticmethod
    def get_bounding_box(preds):
        res = []
        kernel = np.ones((5, 5), dtype=np.uint8)
        for pred in preds:
            opening = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnt, hie = zip(*filter(lambda c: c[1][3] == -1, zip(contours, *hierarchy)))  # remove inner box
            res.append((cnt, hie))
        return res

    @staticmethod
    def show_bbox(pred_mask, cnts):
        img = np.zeros((325, 325, 3), np.uint8)
        for cnt in cnts:
            # min_area_rectangle
            min_rect = cv2.minAreaRect(cnt)
            min_rect = np.int0(cv2.boxPoints(min_rect))
            cv2.drawContours(img, [min_rect], 0, (0, 255, 0), 2)  # green

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def encode_json(bbox):
        res = dict()
        return json.dumps(res)

    def get_result(self):
        # preds_pix = self.pixelize(self.preds)
        bbox = self.get_bounding_box(self.preds)
        res = self.encode_json(bbox)
        return res
