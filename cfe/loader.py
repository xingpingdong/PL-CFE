# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class MultiCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_k):
        self.base_transform = base_transform
        self.n_k = n_k

    def __call__(self, x):
        q = self.base_transform(x)
        if self.n_k==1:
            k = self.base_transform(x)
        else:
            k=[]
            for kk in range(self.n_k):
                k.append(self.base_transform(x))
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
