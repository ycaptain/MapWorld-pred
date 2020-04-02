import argparse
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from utils import prepare_device
from parse_config import ConfigParser
import model as module_arch
import utils.crf as postps_crf


class DemoMain:
    def __init__(self, args, config):
        self.args = args.parse_args()
        self.config = config

        self.logger = config.get_logger('demo')
        self.device, device_ids = prepare_device(self.logger, config['n_gpu'])

        torch.set_grad_enabled(False)
        self.model = config.init_obj('arch', module_arch)
        self.logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.postprocessor = None
        if 'postprocessor' in config["tester"]:
            module_name = config["tester"]['postprocessor']['type']
            module_args = dict(config["tester"]['postprocessor']['args'])
            self.postprocessor = getattr(postps_crf, module_name)(**module_args)

        self.classes = self.get_classtable()

        self.demo_dir = config.save_dir / "demo"
        self.demo_dir.mkdir(parents=True, exist_ok=True)

    def get_classtable(self):
        if "data_loader" in self.config.config:
            if "args" in self.config["data_loader"]:
                if "classes" in self.config["data_loader"]["args"]:
                    return [item for sublist in self.config["data_loader"]["args"]["classes"] for item in sublist.keys()]
        return None

    def preprocessing(self, image):
        w = h = self.config["tester"]["crop_size"]
        img = image.resize((w, h), resample=Image.NEAREST)
        img = np.asarray(img)
        t_img = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32)).float().unsqueeze(0)
        t_img = t_img.to(self.device)
        return t_img, img.astype(np.uint8)

    def inference(self, image, raw_image=None, postprocessor=None):
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = self.model(image)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs)

        #labelmap = np.argmax(probs, axis=0)

        return probs

    def single(self, path):
        image = Image.open(path)
        image, raw_image = self.preprocessing(image)
        probs = self.inference(image, raw_image, self.postprocessor)
        return probs, raw_image

    def show(self, probs, raw_img, in_name=("", "")):
        rows = np.floor(np.sqrt(len(probs) + 1))
        cols = np.ceil((len(probs) + 1) / rows)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(rows, cols, 1)
        ax.set_title(in_name[0]+"/"+in_name[1])
        ax.imshow(raw_img)
        ax.axis("off")

        for i, label in enumerate(probs):
            ax = plt.subplot(rows, cols, i + 2)
            if self.classes is not None:
                ax.set_title(self.classes[i])
            ax.imshow(raw_img)
            ax.imshow(label.astype(np.float32), alpha=0.5)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.savefig(self.demo_dir / (str(in_name[0])+"_"+str(in_name[1])))

    def recur_path(self, file_list):
        for img_path in file_list:
            img_path = Path(img_path)
            if os.path.isfile(img_path):
                probs, raw_image = self.single(img_path)
                # img = Image.fromarray(res.astype(np.uint8))
                # Image.eval(img, lambda a: 255 if a >= 1 else 0).show()
                self.show(probs, raw_image, (os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path)))
            elif os.path.isdir(img_path):
                self.recur_path([img_path / f for f in os.listdir(img_path)])
            else:
                self.logger.critical("Cannot open image path: " + str(img_path))

    def main(self):
        self.recur_path(self.args.image)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Demo')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--image', nargs='+', required=True,
                      help='path to image to be processed')

    config = ConfigParser.from_args(args)
    m = DemoMain(args, config)
    m.main()
