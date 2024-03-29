from . import SpaceNetDataLoader
import cv2


class SpaceNetRoadDataLoader(SpaceNetDataLoader):

    def trsfm_label(self, label):
        w, h = label.size
        image = self._augmentation(label, h, w)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_size, self.dilate_size))
        dilate = cv2.dilate(image, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(dilate, contours, -1, self.ignore_label, thickness=self.ignore_size)
        return dilate

    def __init__(self, data_name, data_dir, batch_size, data_type, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True, scales=None,
                 crop_size=325, ignore_label=255, dilate_size=15, ignore_size=3):
        super().__init__(data_name, data_dir, batch_size, data_type, shuffle, validation_split,
                         num_workers, training, scales, crop_size, ignore_label)
        self.dilate_size = dilate_size
        self.ignore_size = ignore_size
