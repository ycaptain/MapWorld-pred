import os
import threading
import math
import hashlib
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from pathlib import Path

from utils.seg_opt import SegmentOutputUtil


class SegPredThread(threading.Thread):
    from .server import ServerMain

    def __init__(self, srv: ServerMain, imgs, metas, target: Path):
        threading.Thread.__init__(self)
        self.srv = srv

        self.imgs = imgs
        self.metas = metas
        self.target = target
        torch.set_grad_enabled(False)

    def inference(self, image, raw_image=None, postprocessor=None):
        _, _, H, W = image.shape
        logits = self.srv.model(image)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        probs = probs.detach().cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs)

        return probs

    def single(self, path):
        save_name = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
        image = Image.open(path)
        W, H = image.size
        ps, cp = self.srv.prescale, self.srv.crop_size
        if ps != 1.0:
            image = image.resize((int(W * ps), int(H * ps)), resample=Image.NEAREST)
            W, H = image.size
        image = np.asarray(image).astype(np.uint8)
        num_h, num_w = math.ceil(H / cp), math.ceil(W / cp)

        for i in range(0, num_h):
            for j in range(0, num_w):
                crop_img = image[i * cp:(i + 1) * cp,
                           j * cp:(j + 1) * cp]
                t_img = torch.from_numpy(np.moveaxis(crop_img, -1, 0).astype(np.float32)).float().unsqueeze(0)
                t_img = t_img.to(self.srv.device)
                # TODO: support batch_size load
                probs = self.inference(t_img, crop_img, self.srv.postprocessor)

                probs[1] *= 255
                res = Image.fromarray(probs[1].astype(np.uint8))
                fname = "{}_{}_{}_{}.png".format(save_name, self.srv.cfg["name"], i, j)
                res.save(self.target / fname)
        return "{}_{}".format(save_name, self.srv.cfg["name"]), num_h, num_w

    def run(self):
        for img_path in self.imgs:
            if os.path.isfile(img_path):
                fname, num_h, num_w = self.single(img_path)
                for i in range(0, num_h):
                    for j in range(0, num_w):
                        opt_util = SegmentOutputUtil(
                            SegmentOutputUtil.load_img(self.target / "{}_{}_{}.png".format(fname, i, j)))
                        # print(opt_util.get_result())
                        # TODO: Add road jsonify support
                        json_path = str(self.target / "{}_{}_{}.json".format(fname, i, j))
                        with open(json_path, 'w') as f:
                            f.write(opt_util.get_result())
                            # TODO: Notify result
            else:
                self.srv.logger.critical("Cannot open image path: " + str(img_path))

            # TODO: Notify progress
