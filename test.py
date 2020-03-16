import argparse
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image

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
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            img = output.cpu().numpy()[0, 1]
            img = Image.fromarray(img)
            img.save("saved/test/"+str(i)+".tif")

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


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
