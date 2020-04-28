from torch.utils.data import ConcatDataset

from base import BaseDataLoader
from .spacenet_set import SpaceNetDataset
import os.path
import random
from PIL import Image
import numpy as np
import cv2


class SpaceNetDataLoader(BaseDataLoader):

    def _augmentation(self, img, h, w):
        # Scaling
        if self.scales is not None and self.cur_scale is None:
            self.cur_scale = random.choice(self.scales)
            h, w = (int(h * self.cur_scale), int(w * self.cur_scale))
            self.start_h = random.randint(0, max(h - self.crop_size, 0))
            self.start_w = random.randint(0, max(w - self.crop_size, 0))
        else:
            h, w = (int(h * self.cur_scale), int(w * self.cur_scale))
            self.cur_scale = None
        img = img.resize((w, h), resample=Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)

        # Padding
        cp = self.crop_size
        pad_h = (cp - (h % cp)) % cp
        pad_w = (cp - (w % cp)) % cp
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }

        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, value=0, **pad_kwargs)

        # Cropping
        end_h = self.start_h + self.crop_size
        end_w = self.start_w + self.crop_size
        image = img[self.start_h:end_h, self.start_w:end_w]

        return image

    def trsfm(self, image):
        w, h = image.size
        return self._augmentation(image, h, w)

    def trsfm_label(self, label):
        w, h = label.size
        return self._augmentation(label, h, w)

    def __init__(self, data_name, data_dir, batch_size, data_type, shuffle=True, validation_split=0.0,
                 num_workers=1,
                 training=True, scales=None, crop_size=325, ignore_label=255):
        self.data_type = data_type
        ds = list()
        for dn in data_name:
            ds.append(SpaceNetDataset(root=os.path.join(data_dir, dn),
                                      train=training,
                                      transform=self.trsfm,
                                      target_transform=self.trsfm_label,
                                      data_type=self.data_type))
        self.cur_scale = None
        self.scales = scales
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.dataset = ConcatDataset(ds)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def get_dataset(self):
        return self.dataset
