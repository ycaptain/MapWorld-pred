import argparse
import collections
import torch
import numpy as np
import os
import sys

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

import trainer as trains
from parse_config import ConfigParser
from utils import init

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    data_loader, model, criterion, metrics = init(config)

    if hasattr(data_loader, "split_validation"):
        valid_data_loader = data_loader.split_validation()
    else:
        valid_data_loader = None
    trainer = config.init_obj('trainer', trains, model, criterion, metrics,
                              config=config,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--ec', '--epoch_count'], type=int, target='arch;args;epoch_count')
    ]
    config = ConfigParser.from_args(args, options)
    config.init_log()
    main(config)
