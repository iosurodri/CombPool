import torch
from src.layers.comb_pool import ChannelwiseCombPool2d, GatedCombPool2d


def pickPoolLayer(pool_type):

    available_poolings = {
        'max': torch.nn.MaxPool2d,
        'avg': torch.nn.AvgPool2d,
        'channelwise': ChannelwiseCombPool2d,
        'gated': GatedCombPool2d
    }
    if pool_type not in available_poolings.keys():
        raise Exception('Provided pool type is not available: Must be one of {}'.format(available_poolings.keys()))
    return available_poolings[pool_type]
