import unittest
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import make_grid

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from data_loader.spacenet_loader import SpaceNetDataLoader


class SpaceNetTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = SpaceNetDataLoader(
            data_name="AOI_3_Paris_Train",
            data_dir="data/FYPData/spacenet",
            batch_size=100,
            shuffle=False,
            validation_split=0.0,
            num_workers=0,
            colors=[
                {"ignore": [0, 0, 0]},
                {"building": [255, 0, 0]},
                {"road": [0, 0, 255]}
            ],
            scales=[0.5, 0.75, 1.0, 1.25, 1.5]
        )
        print(cls.dataset.dataset)

    def test_preview(self):
        kwargs = {"nrow": 10, "padding": 40}
        for i, (images, labels) in enumerate(self.dataset):
            if i == 0:
                # mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
                # images += mean.expand_as(images)
                image = make_grid(images, pad_value=-1, **kwargs).numpy()
                image = np.transpose(image, (1, 2, 0))
                mask = np.zeros(image.shape[:2])
                mask[(image != -1)[..., 0]] = 255
                image = np.dstack((image, mask)).astype(np.uint8)

                labels = labels[:, np.newaxis, ...]
                label = make_grid(labels, pad_value=255, **kwargs).numpy()
                label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
                label = cm.jet_r(label_ / 3.0) * 255
                mask = np.zeros(label.shape[:2])
                label[..., 3][(label_ == 255)] = 0
                label = label.astype(np.uint8)

                tiled_images = np.hstack((image, label))
                # cv2.imwrite("./docs/datasets/voc12.png", tiled_images)
                plt.figure(figsize=(40, 20))
                plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])), aspect='auto')
                plt.show()
                break


if __name__ == '__main__':
    unittest.main()
