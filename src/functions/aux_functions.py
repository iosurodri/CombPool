import torch
from src.layers.comb_pool import ChannelwiseCombPool2d, GatedCombPool2d
from src.layers.penalty_pool import PenaltyPool2d

def pickPoolLayer(pool_type):

    available_poolings = {
        'max': torch.nn.MaxPool2d,
        'avg': torch.nn.AvgPool2d,
        'channelwise': ChannelwiseCombPool2d,
        'gated': GatedCombPool2d,
        'penalty_norm1': lambda kernel_size, stride=None, padding=0, dilation=1, num_channels=1,
                 aggregations=['avg', 'max']: PenaltyPool2d(kernel_size, stride, padding, dilation, num_channels, aggregations, distance='norm1'),
        'penalty_norm2': lambda kernel_size, stride=None, padding=0, dilation=1, num_channels=1,
                 aggregations=['avg', 'max']: PenaltyPool2d(kernel_size, stride, padding, dilation, num_channels, aggregations, distance='norm2'),
        'penalty_norm3': lambda kernel_size, stride=None, padding=0, dilation=1, num_channels=1,
                 aggregations=['avg', 'max']: PenaltyPool2d(kernel_size, stride, padding, dilation, num_channels, aggregations, distance='norm3'),
    }
    if pool_type not in available_poolings.keys():
        raise Exception('Provided pool type is not available: Must be one of {}'.format(available_poolings.keys()))
    return available_poolings[pool_type]
