import torch
import torch.nn.functional as F

class StochasticPool2DLayer(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if type(stride) == int:
            stride = (stride, stride)
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, tensor, training=False):

        # TODO: Stochastic Pooling requires all positive values, so a ReLU activation is aplied to the input:
        tensor = F.relu(tensor)

        if isinstance(self.padding, list) or isinstance(self.padding, tuple):
            tensor = F.pad(tensor, self.padding)
        # 1.-Extract patches of kernel_size from tensor:
        # ToDo: Debug
        tensor = tensor.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        if self.training:
            # 2.-Turn each one of those 2D patches into a 1D vector:
            original_dims = tensor.shape[0:4]
            tensor = tensor.reshape(-1, self.kernel_size[0] * self.kernel_size[1])
            sum_tensor = tensor.sum(dim=-1, keepdims=True)
            sum_tensor[sum_tensor == 0] = 1
            probabilities = tensor / sum_tensor
            probabilities[probabilities.sum(dim=1) <= 0] = 1
            # probabilities[probabilities <= 0] = 1
            selected_indices = torch.multinomial((probabilities[probabilities.sum(dim=1) > 0]), 1).squeeze()
            tensor = torch.gather(tensor, dim=1, index=selected_indices.unsqueeze(1))
            tensor = tensor.reshape(original_dims)
        else:
            tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                                     self.kernel_size[0] * self.kernel_size[1]))
            sum_tensor = tensor.sum(dim=-1, keepdims=True)
            sum_tensor[sum_tensor == 0] = 1
            probabilities = tensor / sum_tensor
            tensor = (tensor * probabilities).sum(dim=-1, keepdims=False)
        return tensor