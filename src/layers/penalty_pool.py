import torch
import torch.nn as nn
import torch.nn.functional as F

import src.functions.aggregation_functions as aggr_funcs
import src.functions.distance_functions as dist_funcs


class PenaltyPool2d(torch.nn.Module):

    available_distances = {
        'norm1': lambda x, y: dist_funcs.norm_distance(x, y, n=1),
        'norm2': lambda x, y: dist_funcs.norm_distance(x, y, n=2),
        'norm3': lambda x, y: dist_funcs.norm_distance(x, y, n=3),
        # 'learnt': lambda x, y, phi=2: nPow_distance(x, y, n=phi)  # NOTE: Impossible to learn the parameter phi as is
    }

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, num_channels=1,
                 aggregations=['avg', 'max'], distance='norm2'):
        # INPUTS:
        # NOTE: aggregations expects a one element list to show a similar implementation than Conv2dGeneric.__init__()
        super().__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        if len(aggregations) == 1:
            print('Warning: Number of chosen aggregations is 1. Make sure this is the intended behaviour.')
        self.aggregations = []
        for aggr in aggregations:
            self.aggregations.append(aggr_funcs.choose_aggregation(aggr))        
        if distance not in self.available_distances.keys():
            raise Exception('Distance {} not available for generic_distance calculation: '.format(distance))
        else:
            self.dist_name = distance
            self.dist = self.available_distances[distance]
        # TODO: Punto 3:
        # if self.dist_name == 'learnt':
        #     self.weight = torch.nn.Parameter(torch.ones([1], dtype=torch.float) * 2)

    def forward(self, input):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            input = F.pad(input, self.padding)
        # 1.-Unfold the values of each patch to be aggregated
        tensor = input.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                 self.kernel_size[0] * self.kernel_size[1]))
        # 2.-Compute reduction based on the chosen functions:

        out_dim = list(tensor.shape)
        out_dim[-1] = 1
        out_dim.insert(0, len(self.aggregations))

        aggregated_values = tensor.new_zeros(out_dim)
        for idx, aggregation in enumerate(self.aggregations):
            aggregated_values[idx, :] = aggregation(tensor, dim=-1, keepdim=True)

        # TODO: Punto 3:
        # if self.dist_name == 'learnt':
        #     distances = self.dist(tensor.unsqueeze(0), aggregated_values, self.weight)
        # else:
        distances = self.dist(tensor.unsqueeze(0), aggregated_values)
        # TODO: Punto 3: 
        # Calcular la ponderacion de cada agregacion en base al resultado de la Penalty
        indices = distances.argmin(dim=0).unsqueeze(0)
        output_tensor = torch.gather(aggregated_values, 0, indices)
        
        return output_tensor.squeeze()
