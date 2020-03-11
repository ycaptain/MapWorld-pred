import torch.nn.functional as F
from base.base_loss import LossFunction


class MnistLost(LossFunction):

    def build_loss(self):
        return self.nll_loss

    def nll_loss(self, output, target):
        return F.nll_loss(output, target)
