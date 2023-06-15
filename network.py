import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.normalize(x)
        return x

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
# from torchmeta.modules.utils import self.get_subdict



# def conv_block(in_channels, out_channels, **kwargs):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, **kwargs),
#         nn.BatchNorm2d(out_channels, momentum=1.,
#             track_running_stats=False),
#         nn.ReLU(),
#         nn.MaxPool2d(2)
#     )
#
# class MetaConv(nn.Module):
#     """4-layer Convolutional Neural Network architecture from [1].
#
#     Parameters
#     ----------
#     in_channels : int
#         Number of channels for the input images.
#
#     out_features : int
#         Number of classes (output of the model).
#
#     hidden_size : int (default: 64)
#         Number of channels in the intermediate representations.
#
#     feature_size : int (default: 64)
#         Number of features returned by the convolutional head.
#
#     References
#     ----------
#     .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
#            for Fast Adaptation of Deep Networks. International Conference on
#            Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
#     """
#     def __init__(self, in_channels, out_features=5, hidden_size=64, feature_size=64):
#         super(MetaConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_features = out_features
#         self.hidden_size = hidden_size
#         self.feature_size = feature_size
#
#         self.features = nn.Sequential(OrderedDict([
#             ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
#                                   stride=1, padding=1, bias=True)),
#             ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
#                                   stride=1, padding=1, bias=True)),
#             ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
#                                   stride=1, padding=1, bias=True)),
#             ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
#                                   stride=1, padding=1, bias=True))
#         ]))
#         # self.classifier = MetaLinear(feature_size, out_features, bias=True)
#
#     def forward(self, inputs, params=None):
#         features = self.features(inputs)
#         features = features.view((features.size(0), -1))
#         # logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
#         return features


def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

class MetaConv(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features=5, hidden_size=64, feature_size=64):
        super(MetaConv, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        # self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        # logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return features

def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConv(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)

def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConv(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)