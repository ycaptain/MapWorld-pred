import cv2
import numpy as np
import os


class GeoJsonRender:

    img = None

    def __init__(self):
        pass

    def create(self, prop):
        self.img = np.zeros(prop, dtype="uint8")

    def draw_item(self, item):
        pass

    def save(self, filename, path):
        cv2.imwrite(os.path.join(path, filename), self.img)

    def preview(self):
        cv2.imshow('Preview', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
