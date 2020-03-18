import argparse
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.nn.functional as F

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

import data_loader as ds
import model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', ds)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = config.init_obj('loss', module_arch.loss_entry).build_loss()
    metric_fns = [getattr(module_arch.metric_entry, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    j=0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            labmap = inference(output, 325, 325)
            # img = Image.fromarray(labmap.astype(np.uint8))
            # img.save("saved/test/"+str(i)+".tif")
            Image.fromarray(np.moveaxis(data.cpu().numpy().astype(np.uint8)[0], 0, -1)).show()
            tmp = Image.fromarray(labmap.astype(np.uint8))
            Image.eval(tmp, lambda a: 255 if a >= 1 else 0).show()
            if j == 10:
                break
            j+=1
            # j = 0
            # for label in output:
            #     label = label.float().cpu().numpy()
            #     label = Image.fromarray(label.astype(np.uint8))
            #     label.save("saved/test/"+str(j)+".tif")
            #     j += 1

            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


def inference(logits, H, W, raw_image=None, postprocessor=None):
    #_, _, H, W = image.shape

    # Image -> Probability map
    # logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
