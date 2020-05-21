import argparse
import os
import sys

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

import trainer as trains
from utils import init
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    data_loader, model, criterion, metrics = init(config)

    tester = config.init_obj('trainer', trains, model, criterion, metrics, config, data_loader, valid_data_loader=None)
    total_loss, total_metrics = tester.test()

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update(total_metrics)
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    config.init_log()
    main(config)
