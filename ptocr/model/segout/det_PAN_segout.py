# -*- coding:utf-8 _*-
"""
@graduation project
@School of Computer Science and Engineering, Beihang University
@author  : hyc
@File    : det_PAN_segout.py
@contact : hyc2026@yeah.net
"""
import torch.nn as nn
from ..CommonFunction import upsample


class SegDetector(nn.Module):
    def __init__(self, inner_channels=128, classes=6):
        super(SegDetector, self).__init__()
        self.binarize = nn.Conv2d(inner_channels, classes, 1, 1, 0)

    def forward(self, x, img):
        x = self.binarize(x)
        x = upsample(x, img)
        if self.training:
            pre_batch = dict(pre_text=x[:, 0])
            pre_batch['pre_kernel'] = x[:, 1]
            pre_batch['similarity_vector'] = x[:, 2:]
            return pre_batch
        return x
