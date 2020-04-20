from base import BaseTrainer
from tqdm import tqdm

from model import CycleGANOptions, CycleGANModel


class CycleGANTrainer(BaseTrainer):
    def __init__(self, model, criterion, metrics, config, data_loader, valid_data_loader=None, len_epoch=None):
        super().__init__(None, None, None, None, config)
        self.data_loader = data_loader
        if not isinstance(model, CycleGANOptions):
            raise NotImplementedError("Model should be configured by CycleGANOptions")
        self.model = CycleGANModel(model)
        self.model.set_input(model)

    def _train_epoch(self, epoch):
        tbar = tqdm(self.data_loader)
        for i, data in enumerate(tbar):
            self.model.set_input(data)
            self.model.optimize_parameters()

    def test(self):
        pass
