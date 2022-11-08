#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-04-25 12:23
# @Author  : Jianming Ip
# @Site    :
# @File    : RSC_gananstyle.py
# @Company : VMC Lab in Peking University


import random

import torch
import torch.nn as nn

DEFAULT_P = 0.01

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, p=0.01, eps=1e-5):
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        permute = random.random() < self.p
        if permute and self.training:
            perm_indices = torch.randperm(x.size()[0])
        else:
            return x
        size = x.size()
        N, C, H, W = size
        if (H, W) == (1, 1):
            print('encountered bad dims')
            return x
        return adaptive_instance_normalization(x, x[perm_indices], self.eps)

    def extra_repr(self) -> str:
        return 'p={}'.format(
            self.p
        )

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps

    def forward(self, x, mean=None, std=None):
        if self.training:
            size = x.size()
            N, C, H, W = size
            if (H, W) == (1, 1):
                print('encountered bad dims')
                return x
            return adaptive_instance_normalization_by_vector(x, mean.view(N, C, 1, 1), std.view(N, C, 1, 1), self.eps)
        else:
            return x

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = torch.sqrt(feat.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat.detach(), eps)
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adaptive_instance_normalization_by_vector(content_feat, style_mean, style_std, eps=1e-5):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)