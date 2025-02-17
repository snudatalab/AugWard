import inspect
import torch.nn as nn

from loss.celoss import SoftCELoss
from loss.target import TargetLoss

class Loss(object):
    def __init__(self, loss=None, **kwargs):
        super().__init__()
        if loss is None:
            raise RuntimeError("Loss w/o type")
        else:
            self.loss_name = loss
            loss_class = eval(self.loss_name)
            parameters = inspect.signature(loss_class).parameters
            args = {k: v for k, v in kwargs.items() if k in parameters}
            self.loss = loss_class(**args)

    def __call__(self, input, target):
        return self.loss(input, target)


        

