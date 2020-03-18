from base import BaseDataLoader
from .spacenet_set import SpaceNetDataset
import os.path
import random
from PIL import Image
import numpy as np


class SpaceNetDataLoader(BaseDataLoader):

    def _augmentation(self, img, h, w):
        # Scaling
        if self.scales is not None and self.cur_scale is None:
            self.cur_scale = random.choice(self.scales)
            scale_factor = self.cur_scale
        else:
            scale_factor = self.cur_scale
            self.cur_scale = None
        if scale_factor is not None:
            h, w = (int(h * scale_factor), int(w * scale_factor))
            img = img.resize((w, h), resample=Image.NEAREST)
            img = np.asarray(img, dtype=np.uint8)

        # # Crop
        # pad_h = max(self.crop_size - h, 0)
        # pad_w = max(self.crop_size - w, 0)
        # pad_kwargs = {
        #     "top": 0,
        #     "bottom": pad_h,
        #     "left": 0,
        #     "right": pad_w,
        #     "borderType": cv2.BORDER_CONSTANT,
        # }
        # if pad_h > 0 or pad_w > 0:
        #     image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
        #     label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = img[start_h:end_h, start_w:end_w]

        return image

    def trsfm(self, image):
        w, h = image.size
        return self._augmentation(image, h, w)

    def trsfm_label(self, label):
        w, h = label.size
        return self._augmentation(label, h, w)

    def __init__(self, data_name, data_dir, batch_size, colors=None, shuffle=True, validation_split=0.0, num_workers=1,
                 training=True, scales=None, crop_size=325):
        if colors is None:
            colors = [{"ignore": [0, 0, 0]}, {"building": [255, 0, 0]}, {"road": [0, 0, 255]}]
        self.data_name = data_name
        self.data_dir = data_dir
        self.colors = colors
        self.cur_scale = None
        self.scales = scales
        self.crop_size = crop_size
        self.dataset = SpaceNetDataset(root=os.path.join(self.data_dir, self.data_name),
                                       colors=colors,
                                       train=training,
                                       transform=self.trsfm,
                                       target_transform=self.trsfm_label)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
