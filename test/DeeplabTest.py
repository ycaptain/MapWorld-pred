import unittest
from pathlib import Path
import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)

from parse_config import ConfigParser
import data_loader.spacenet_loader as module_data


class DeeplabTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config = ConfigParser(ConfigParser.from_file("test/configs/deeplab.json"))

        cls.config = config
        # cls.datadir = Path(config['data_loader']['args']['data_dir'])
        # cls.data_name = config['data_loader']['args']['data_name']

    def run(self):
        config = self.config
        logger = config.get_logger('train')
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        logger.info(model)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

        trainer.train()


if __name__ == '__main__':
    unittest.main()
