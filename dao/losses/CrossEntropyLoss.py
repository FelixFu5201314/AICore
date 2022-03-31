import numpy as np
import torch
import torch.nn as nn

from dao.register import Registers


@Registers.losses.register
def CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean'):
        weight = torch.from_numpy(np.array(weight)).float()
        CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        return CE


