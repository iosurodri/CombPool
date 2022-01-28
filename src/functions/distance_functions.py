import torch
import torch.nn as nn
    
### Distances:

def abs_distance(x, y):
    return torch.abs(x - y)

def sqrt_distance(x, y):
    return torch.sqrt(x*x - y*y)

def nPow_distance(x, y, n=2):
    return torch.pow(torch.pow(x, n), torch.pow(y, n), 1.0/n)


'''Generic distance implementation'''

def generic_distance(tensor, aggregated_values, distance='abs_dist', reduction='sum'):
    
    available_distances = {
        'abs_dist': abs_distance,
        'sqrt_dist': sqrt_distance,
        '3pow_dist': lambda x, y: nPow_distance(x, y, n=3)
    }

    available_reductions = {
        'sum': torch.sum,
    }

    if distance not in available_distances.keys():
        raise Exception('Distance {} not available for generic_distance calculation: '.format(distance))
    else:
        dist = available_distances[distance]
    if reduction not in available_reductions.keys():
        raise Exception('Reduction {} not available for generic_distance calculation: '.format(reduction))
    else:
        reduct = available_reductions[reduction]

    # Calculate distances as the reduction of a distance function:
    distances = reduct(dist(tensor.unsqueeze(0), aggregated_values), dim=-1, keepdim=True)
    # Get the indices of values that minimize the penalty function:
    indices = distances.argmin(dim=0).unsqueeze(0)
    # Get the values of aggregated_values that minimize the penalty function:
    output_values = torch.gather(aggregated_values, 0, indices)
    return output_values
    

''' These implementations are deprecated.
If implementation via generic_distance extensions is hard, the penalty can be implemented directly.
See the following functions as examples:
'''

def old_natural_distance(tensor, aggregated_values):
    distances = torch.abs(tensor.unsqueeze(0) - aggregated_values)  # Compute the absolute difference between all values
    distances = distances.sum(dim=-1, keepdim=True)  # Compute the mean of all values
    indices = distances.argmin(dim=0).unsqueeze(0)
    output_values = torch.gather(aggregated_values, 0, indices)
    return output_values


def old_euclidean_distance(tensor, aggregated_values):
    distances = torch.sqrt((tensor * tensor).unsqueeze(0) - (aggregated_values * aggregated_values))  # Compute the absolute difference between all values
    distances = distances.sum(dim=-1, keepdim=True)  # Compute the mean of all values
    indices = distances.argmin(dim=0).unsqueeze(0)
    output_values = torch.gather(aggregated_values, 0, indices)
    return output_values



def choose_distance(dist_name):
    '''Returns one of the available functions if the key has been provided correctly
    :param aggr_name: Name of the distance function to be returned.
    :type aggr_name: str
    :return: Distance function (defined either in this module or in torch)
    '''
    if dist_name not in available_distances.keys():
        raise Exception('Unknown distance function. For a list of available functions, use list_available_distances().')
    return available_distances[dist_name]


available_distances = {
    'natural': lambda tensor, aggregated_values: generic_distance(tensor, aggregated_values, distance='abs_dist', reduction='sum'),
    'euclidean': lambda tensor, aggregated_values: generic_distance(tensor, aggregated_values, distance='sqrt_dist', reduction='sum'),
    'old_natural': old_natural_distance,
    'old_euclidean': old_euclidean_distance,
}


''' IMPLEMENTATION AS MODULE:
Implementation that allows for parameterization of the penalty function:
'''

class param_distance(nn.Module):
    
    available_distances = {
        'abs_dist': abs_distance,
        'sqrt_dist': sqrt_distance,
        '3pow_dist': lambda x, y: nPow_distance(x, y, n=3)
    }
    available_reductions = {
        'sum': torch.sum,
    }

    def __init__(self, distance='abs_dist', reduction='sum'):

        if distance not in self.available_distances.keys():
            raise Exception('Distance {} not available for generic_distance calculation: '.format(distance))
        else:
            self.dist = self.available_distances[distance]
        if reduction not in self.available_reductions.keys():
            raise Exception('Reduction {} not available for generic_distance calculation: '.format(reduction))
        else:
            self.reduct = self.available_reductions[reduction]
        # Example parameter to use:
        self.weight = self.Parameter(torch.ones([1], dtype=torch.float) * 2)   

    def forward(self, tensor, aggregated_values):
        # NOTE: self.weight is not used in this example function.
        distances = self.dist(tensor.unsqueeze(0), aggregated_values)
        distances = self.reduct(distances, dim=-1, keepdim=True)
        indices = distances.argmin(dim=0).unsqueeze(0)
        output_values = torch.gather(aggregated_values, 0, indices)
        return output_values