import torch
from src.layers.comb_pool import *
from src.layers.external_layers import *


def pickPoolLayer(pool_type):

    available_poolings = {
        'max': torch.nn.MaxPool2d,
        'avg': torch.nn.AvgPool2d,
        'channelwise': ChannelwiseCombPool2d,
        'gated': GatedCombPool2d,
        'stochastic': StochasticPool2DLayer,
        # 'constant': ConstantCombPool2d,
        # 'test1': test1ConstantCombPool2d,
        # 'test2': test2ConstantCombPool2d,
        # 'test3': test3ConstantCombPool2d,
        # 'test4': test4ConstantCombPool2d,
        # 'test5': test5ConstantCombPool2d,
        # 'test6': test6ConstantCombPool2d,
    }
    if pool_type not in available_poolings.keys():
        raise Exception('Provided pool type is not available: Must be one of {}'.format(available_poolings.keys()))
    return available_poolings[pool_type]


def pickGlobalPoolLayer(global_pool_type):

    available_poolings = {
        'max': torch.nn.AdaptiveMaxPool2d,
        'avg': torch.nn.AdaptiveAvgPool2d,
        'channelwise': ChannelwiseGlobalCombPool2d,
        'gated': GatedGlobalCombPool2d
    }
    if global_pool_type not in available_poolings.keys():
        raise Exception('Provided global pool type is not available: Must be one of {}'.format(available_poolings.keys()))
    return available_poolings[global_pool_type]
