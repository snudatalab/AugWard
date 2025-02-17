import torch
import torch.nn as nn

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.l2 = nn.MSELoss(reduction="mean")
    
    def forward(self, first, second):
        first_normalized = first / torch.linalg.norm(first)
        second_normalized = second / torch.linalg.norm(second)
        return self.l2(first_normalized, second_normalized)
    
class TargetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L2Norm()

    def forward(self, input, target):
        return self.loss(input, target)