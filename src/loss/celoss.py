import torch
import torch.nn as nn

class SoftCELoss(nn.Module):
    def __init__(self, temp=100, epsilon=1e-10):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.temp = temp
        self.epsilon = epsilon

    def forward(self, input, target):
        if target.ndim == 1:
            return self.ce_loss(input, target)
        elif input.size() == target.size():
            input = self.log_softmax(input / self.temp) + self.epsilon
            target = self.softmax(target / self.temp)
            return self.kl_loss(input, target) * self.temp
        else:
            raise ValueError(input.size(), target.size())