import torch.nn
import torch.nn.functional as F

import src.functions.aggregation_functions as aggr_funcs

class CombPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, num_channels=1,
                 aggregations=['avg', 'max'], coefficient_type='channelwise'):
        # INPUTS:
        # coefficient_type: str -> Indicates the way in which coefficients are to be computed
        #   coefficient_type == 'channelwise' -> A different coefficient for each channel of the layer
        #   coefficient_type == 'gated' -> A different coefficient will be computed for each window of the input, through a linear regression.
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
        if coefficient_type == 'channelwise':
            # One value by channel (depth dimension)
            self.coefficients = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(1, num_channels, 1, 1)) for i in range(len(self.aggregations))])
        elif coefficient_type == 'gated':
            # A series of weights which help to generate a different alpha value for each value of a window            
            self.coefficients = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.rand(1, num_channels, 1, 1, kernel_size*kernel_size)) for i in range(len(self.aggregations))]
            )
        else:
            raise Exception('Wrong option specified for coefficient_type. Must be one of "channelwise" or "gated"')
        self.coefficient_type = coefficient_type        
        

    def forward(self, input):
        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            input = F.pad(input, self.padding)
        if self.coefficient_type == 'channelwise':
            new_coefficients = [x**2 for x in self.coefficients]
        if self.coefficient_type == 'gated':
            # This learning method indicates that the value of alpha is computed as the product of a window of values
            # with each patch of the image, and applying a sigmoid function to the output, in order to get a value in (0, 1):
            # 1.-Extract patches of kernel_size from tensor:
            tensor = input.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
            # 2.-Turn each one of those 2D patches into a 1D vector:
            tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                     self.kernel_size[0] * self.kernel_size[1]))
            # 3.-Compute the value of alpha (depends on weights, bias and each of the windows to be applied to):
            new_coefficients = list(map(lambda x: torch.sigmoid(torch.sum(tensor * x, dim=-1, keepdim=False)), self.coefficients))
        # 4.-Unfold the values of each patch to be aggregated
        tensor = input.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], self.kernel_size[0] * self.kernel_size[1]))
        # 5.-Compute reduction based on the chosen functions:
        # Generate an auxiliar tensor the size of input, with as many values for the last dimension as groupings to apply
        output_tensor = tensor.new_zeros(tensor.shape[:-1])
        for idx, aggregation in enumerate(self.aggregations):
            output_tensor += new_coefficients[idx] * aggregation(tensor, dim=-1) 
        return output_tensor


class ChannelwiseCombPool2d(CombPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, num_channels=1,
                 aggregations=['avg', 'max']):
        super().__init__(kernel_size, stride, padding, dilation, num_channels, aggregations, coefficient_type='channelwise')


class GatedCombPool2d(CombPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, num_channels=1,
                 aggregations=['avg', 'max']):
        super().__init__(kernel_size, stride, padding, dilation, num_channels, aggregations, coefficient_type='channelwise')