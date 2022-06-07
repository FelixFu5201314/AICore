import numpy as np
import torch
import torch.nn as nn

from dao.register import Registers


@Registers.losses.register
def CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean'):
    if weight is not None:
        weight = torch.from_numpy(np.array(weight))
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)


if __name__ == "__main__":
    # Example of target with class indices
    loss = CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()

    # Example of target with class probabilities
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    output = loss(input, target)
    output.backward()
