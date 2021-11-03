import torch
import numpy as np

### We precompute tensors for the most common fuzzy measures required in choquet and sugeno integrals:

# Generate fuzzy measure (cardinality) for 2 by 2 windows:
fuzzy_measure_2by2_val = list(range(4, 0, -1))
fuzzy_measure_2by2_val = list(map(lambda x: float(x) / 4, fuzzy_measure_2by2_val))
fuzzy_measure_2by2_val = np.multiply(fuzzy_measure_2by2_val, fuzzy_measure_2by2_val)  # p = 2
# Generate fuzzy measure (cardinality) for 3 by 3 windows:
fuzzy_measure_3by3_val = list(range(9, 0, -1))
fuzzy_measure_3by3_val = list(map(lambda x: float(x) / 9, fuzzy_measure_3by3_val))
fuzzy_measure_3by3_val = np.multiply(fuzzy_measure_3by3_val, fuzzy_measure_3by3_val)  # p = 2
# Generate tensors representing the two most common fuzzy measures
fuzzy_measure_2by2 = torch.tensor(fuzzy_measure_2by2_val, dtype=torch.float, requires_grad=False)
fuzzy_measure_3by3 = torch.tensor(fuzzy_measure_3by3_val, dtype=torch.float, requires_grad=False)
# Check if gpu is being used and create tensors for the fuzzy measures:
if torch.cuda.is_available():
    fuzzy_measure_2by2_cuda = torch.tensor(fuzzy_measure_2by2_val, dtype=torch.float, requires_grad=False).to('cuda:0')
    fuzzy_measure_3by3_cuda = torch.tensor(fuzzy_measure_3by3_val, dtype=torch.float, requires_grad=False).to('cuda:0')

def choose_aggregation(aggr_name):
    '''Returns one of the available functions if the key has been provided correctly
    :param aggr_name: Name of the aggregation function to be returned.
    :type aggr_name: str
    :return: Aggregation function (defined either in this module or in torch)
    '''
    if aggr_name not in available_functions.keys():
        raise Exception('Unknown aggregation function. For a list of available functions, use list_available_functions().')
    return available_functions[aggr_name]

####################
# ORDER STATISTICS #
####################

def min(tensor, keepdim=False, dim=0):
    return torch.min(tensor, keepdim=keepdim, dim=dim)[0]


def max(tensor, keepdim=False, dim=0):
    return torch.max(tensor, keepdim=keepdim, dim=dim)[0]


def median(tensor, keepdim=False, dim=0):
    return torch.median(tensor, keepdim=keepdim, dim=dim)[0]

###################
# FUZZY INTEGRALS #
###################

def sugeno(tensor, fuzzy_measure=None, keepdim=False, dim=0):
    tensor = torch.sort(tensor, dim=dim, descending=False)[0]

    if fuzzy_measure is None:
        if tensor.shape[dim] == 9:
            # Usual convolutional window shape (already pregenerated)
            if tensor.device.type == 'cuda':
                fuzzy_measure = fuzzy_measure_3by3_cuda
            else:
                fuzzy_measure = fuzzy_measure_3by3
        elif tensor.shape[dim] == 4:
            # Usual pooling window shape (already pregenerated)
            if tensor.device.type == 'cuda':
                fuzzy_measure = fuzzy_measure_2by2_cuda
            else:
                fuzzy_measure = fuzzy_measure_2by2
        else:
            # Generate fuzzy measure based in the cardinality of the number of elements to aggregate:
            values_fuzzy_measure = list(range(tensor.shape[dim], 0, -1))
            values_fuzzy_measure = list(map(lambda x: float(x) / tensor.shape[dim], values_fuzzy_measure))
            values_fuzzy_measure = np.multiply(values_fuzzy_measure, values_fuzzy_measure)  # p = 2
            fuzzy_measure = tensor.new_tensor(values_fuzzy_measure)
    tensor = torch.max(torch.min(tensor, fuzzy_measure), dim=dim, keepdim=keepdim)[0]
    return tensor

def dv_sugeno_general(tensor, fuzzy_measure=None, keepdim=False, dim=0):
    tensor = torch.sort(tensor, dim=dim, descending=False)[0]

    if fuzzy_measure is None:
        if tensor.shape[dim] == 9:
            # Usual convolutional window shape (already pregenerated)
            if tensor.device.type == 'cuda':
                fuzzy_measure = fuzzy_measure_3by3_cuda
            else:
                fuzzy_measure = fuzzy_measure_3by3
        elif tensor.shape[dim] == 4:
            # Usual pooling window shape (already pregenerated)
            if tensor.device.type == 'cuda':
                fuzzy_measure = fuzzy_measure_2by2_cuda
            else:
                fuzzy_measure = fuzzy_measure_2by2
        else:
            # Generate fuzzy measure based in the cardinality of the number of elements to aggregate:
            values_fuzzy_measure = list(range(tensor.shape[dim], 0, -1))
            values_fuzzy_measure = list(map(lambda x: float(x) / tensor.shape[dim], values_fuzzy_measure))
            values_fuzzy_measure = np.multiply(values_fuzzy_measure, values_fuzzy_measure)  # p = 2
            fuzzy_measure = tensor.new_tensor(values_fuzzy_measure)
    tensor = torch.sum(torch.mul(tensor, fuzzy_measure), dim=dim, keepdim=keepdim)
    return tensor


available_functions = {
    'min': min,
    'max': max,
    'median': median,
    'sum': torch.sum,
    'avg': torch.mean,
    'sugeno': sugeno,
    'dv_sugeno_general': dv_sugeno_general
}