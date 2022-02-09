import torch
import torch.nn as nn
    
### Distances:

def norm_distance(x, y, n=2):
    # TODO: Check that no nan values are generated:
    # return torch.pow(torch.sum(torch.pow(torch.abs(x - y), n), dim=-1, keepdim=True), 1.0/n)
    return torch.pow(torch.sum(torch.pow(x - y, n), dim=-1, keepdim=True), 1.0/n)
