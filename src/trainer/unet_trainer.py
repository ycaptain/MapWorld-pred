from .deeplab_trainer import DeeplabTrainer
from utils import inf_loop
import numpy as np
import model as module_arch
import torch


class UNetTrainer(DeeplabTrainer):
    def __init__(self, model, criterion, metric_ftns, config, data_loader,
                 valid_data_loader=None, len_epoch=None):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        super(DeeplabTrainer, self).__init__(model, criterion, metric_ftns, optimizer, config)
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', module_arch.lr_entry, optimizer)
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = metric_ftns[0]
        self.valid_metrics = metric_ftns[0]

        self.logit_dir = None
        if config["tester"].get("save_logits", False):
            self.logit_dir = config.save_dir / "logits"
            self.logit_dir.mkdir(parents=True, exist_ok=True)
