import time

from base import BaseTrainer
from tqdm import tqdm

from model import CycleGANOptions, CycleGANModel
from utils.cycgan import Visualizer
from utils.util import DotDict


class CycleGANTrainer(BaseTrainer):
    def __init__(self, model, criterion, metrics, config, data_loader, valid_data_loader="Test", len_epoch=None):
        super().__init__(None, None, None, None, config)
        self.data_loader = data_loader
        if not isinstance(model, CycleGANOptions):
            raise NotImplementedError("Model should be configured by CycleGANOptions")

        model.checkpoints_dir = self.checkpoint_dir
        model.name = config.config['name']
        model.preprocess = data_loader.opt.preprocess
        model.direction = data_loader.opt.direction

        if valid_data_loader is "Test":
            model.isTrain = False
        else:
            model.isTrain = True
        self.start_epoch = model.epoch_count
        self.model = CycleGANModel(model)
        vis_opt = DotDict({
            "isTrain": model.isTrain,
            "display_id": 1,
            "display_server": "http://localhost",
            "display_env": "main",
            "display_port": 8097,
            "display_ncols": 4,
            "display_winsize": 256,
            "checkpoints_dir": self.checkpoint_dir,
            "name": model.name
        })
        self.visualizer = Visualizer(vis_opt)
        if self.continue_train:
            load_suffix = config.resume.name
            self.model.load_networks(load_suffix)

    def _train_epoch(self, epoch):
        tbar = tqdm(self.data_loader)
        for i, data in enumerate(tbar):
            self.model.set_input(data)
            self.model.optimize_parameters()
            # if i % self.log_step == 0:
            #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         self._progress(batch_idx),
            #         loss.item()))
        log = self.model.get_current_losses()
        return log

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            iter_start_time = time.time()
            result = self._train_epoch(epoch)

            losses = self.model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / self.data_loader.opt.batch_size
            self.visualizer.print_current_losses(epoch, 0, losses, t_comp, 0)
            self.visualizer.plot_current_losses(epoch, 0, losses)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            self.model.compute_visuals()
            self.visualizer.display_current_results(self.model.get_current_visuals(), epoch, True)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch % self.save_period == 0:
                print('saving the model at the end of epoch %d' % epoch)
                self.model.save_networks('latest')
                self.model.save_networks(epoch)

    def _resume_checkpoint(self, resume_path):
        self.continue_train = True

    def test(self):
        pass
