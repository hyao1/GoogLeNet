"""
pairwise loss : implement of paper <<Pairwise Confusionfor Fine-Grained Visual Classification >>
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class PCLoss(nn.Module):

    def __init__(self):
        super(PCLoss, self).__init__()

    def forward(self, features):
        features = F.softmax(features, dim=1)
        batch_size = features.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = features[:int(0.5 * batch_size)]
        batch_right = features[int(0.5 * batch_size):]
        loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

        return loss
