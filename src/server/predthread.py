import os
import threading
import math
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
import cv2

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
        probs = F.softmax(logits, dim=1)
        probs = probs.detach().cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            res = list()
            for i, p in zip(raw_image, probs):
                res.append(postprocessor(i, p) * 255)
            return res
        else:
            return probs

    @staticmethod
    def img_iter(img, crop, h, w):
        for i in range(0, h):
            for j in range(0, w):
                crop_img = img[i * crop:(i + 1) * crop,
                           j * crop:(j + 1) * crop]
                t_img = torch.from_numpy(np.moveaxis(crop_img, -1, 0).astype(np.float32)).float().unsqueeze(0)
                yield t_img, crop_img

    def single(self, path):
        save_name = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
        image = Image.open(path).convert('RGB')
        W, H = image.size
        ps, cp = self.srv.prescale, self.srv.crop_size
        if ps != 1.0:
            image = image.resize((int(W * ps), int(H * ps)), resample=Image.NEAREST)
        image = np.asarray(image).astype(np.uint8)

        # Padding to fit for crop_size
        H, W, _ = image.shape
        pad_h = (cp - (H % cp)) % cp
        pad_w = (cp - (W % cp)) % cp
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }

        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)

        H, W, _ = image.shape
        num_h, num_w = math.ceil(H / cp), math.ceil(W / cp)

        count = 0
        t_list = list()
        raw_img_list = list()

        def do_pred(t_list, raw_img_list, count):
            t_list = torch.cat(t_list)

            t_list = t_list.to(self.srv.device)
            probs = self.inference(t_list, raw_img_list, self.srv.postprocessor)

            # save result
            for res, img in zip(probs, raw_img_list):
                fname = "{}_{}_{}.png".format(save_name, self.srv.cfg["name"], count)
                # cv2.imwrite(str(self.target / fname), np.moveaxis(res.astype(np.uint8), 0, -1))
                cv2.imwrite(str(self.target / fname), res.astype(np.uint8)[1])
                # save raw img
                # fname = "{}_{}.png".format(save_name, count)
                # cv2.imwrite(str(self.target / fname), img.astype(np.uint8))
                count += 1
            return count

        for t_img, raw_img in self.img_iter(image, cp, num_h, num_w):
            t_list.append(t_img)
            raw_img_list.append(raw_img)

            if len(t_list) >= self.srv.batch_size:
                count = do_pred(t_list, raw_img_list, count)

                t_list = list()
                raw_img_list = list()

        if len(t_list) != 0:
            do_pred(t_list, raw_img_list, count)

        # return "{}_{}".format(save_name, self.srv.cfg["name"]), num_h, num_w
        return save_name, num_h, num_w

    def gen_meta(self, meta, img):
        width, height = img.shape
        return {
            "origin.x": meta.origin.x,
            "origin.y": meta.origin.y,
            # "pixel_size.x": meta.pixel_size.x,
            # "pixel_size.y": meta.pixel_size.y,
            "w": width,
            "h": height,
            "prescale": self.srv.prescale,
        }

    def run(self):
        count = 0
        total = len(self.imgs)
        for img_path, meta in zip(self.imgs, self.metas):
            if os.path.isfile(img_path):
                # single prediction
                fname, num_h, num_w = self.single(img_path)
                count = 0
                # proc JSON
                for i in range(0, num_h):
                    for j in range(0, num_w):
                        # model output image path
                        label_path = self.target / "{}_{}_{}.png".format(fname, self.srv.cfg["name"], count)
                        img = SegmentOutputUtil.load_img(label_path)
                        t_meta = self.gen_meta(meta, img)
                        t_meta["img_path"] = os.path.abspath(img_path)
                        opt_util = SegmentOutputUtil(img, t_meta, self.srv.cfg["name"])
                        json_path = opt_util.get_result(str(self.target / "{}_{}".format(fname, count)))
                        # print(opt_util.get_result())
                        # # TODO: Add road jsonify support
                        # json_path = str(self.target / "{}_{}.json".format(fname, count))
                        # with open(json_path, 'w') as f:
                        #     f.write(opt_util.get_result())
                        self.srv.send_result(label_path, json_path)
                        count += 1
            else:
                self.srv.logger.critical("Cannot open image path: " + str(img_path))
            count += 1
            # TODO: Notify progress
            self.srv.send_progress(total, count, img_path)
