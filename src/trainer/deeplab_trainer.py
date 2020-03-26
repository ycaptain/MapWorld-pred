import numpy as np
import torch
import os
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop
import model as module_arch
import utils.crf as postps_crf


class DeeplabTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, config, data_loader,
                 valid_data_loader=None, len_epoch=None):
        lr = config["optimizer"]["lr"]
        weight_decay = config["optimizer"]["weight_decay"]
        optimizer = torch.optim.SGD(
            params=[{'params': model.get_1x_lr_params(), 'lr': lr, "weight_decay": weight_decay},
                    {'params': model.get_10x_lr_params(), 'lr': lr * 10, "weight_decay": weight_decay}],
            momentum=config["optimizer"]["momentum"],
        )
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.train_metrics = metric_ftns[0]
        self.valid_metrics = metric_ftns[0]
        self.config = config
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

        self.postprocessor = None

        if 'postprocessor' in config["tester"]:
            module_name = config["tester"]['postprocessor']['type']
            module_args = dict(config["tester"]['postprocessor']['args'])
            self.postprocessor = getattr(postps_crf, module_name)(**module_args)

        self.logit_dir = config.save_dir / "logits"
        self.logit_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        tbar = tqdm(self.data_loader)
        for batch_idx, (img_id, data, target) in enumerate(tbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = dict()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        tbar = tqdm(self.valid_data_loader)
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (img_id, data, target) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (batch_idx + 1)))
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                # Add batch sample into evaluator
                self.valid_metrics.add_batch(target, pred)

        # Fast test during the training
        result = self.valid_metrics.result()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', result["mIoU"], epoch)
        self.writer.add_scalar('val/Acc', result["Acc"], epoch)
        self.writer.add_scalar('val/Acc_class', result["Acc_class"], epoch)
        self.writer.add_scalar('val/fwIoU', result["FWIoU"], epoch)
        print('Validation:')
        print('[Epoch: %d]' % epoch)
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(result["Acc"],
                                                                result["Acc_class"],
                                                                result["mIoU"],
                                                                result["FWIoU"]))
        print('Loss: %.3f' % test_loss)

        return result

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @staticmethod
    def get_params(model, key):
        # For Dilated FCN
        if key == "1x":
            for m in model.named_modules():
                if "layer" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
        # For conv weight in the ASPP module
        if key == "10x":
            for m in model.named_modules():
                if "aspp" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        yield m[1].weight
        # For conv bias in the ASPP module
        if key == "20x":
            for m in model.named_modules():
                if "aspp" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        yield m[1].bias

    def test(self):
        with torch.no_grad():
            tbar = tqdm(self.data_loader)
            for i, (image_ids, data, target) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                for image_id, logit in zip(image_ids, output):
                    filename = os.path.join(self.logit_dir, image_id + ".npy")
                    np.save(filename, logit.cpu().numpy())
