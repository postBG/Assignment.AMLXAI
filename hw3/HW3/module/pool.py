import torch
from torch.nn import functional as F


class MaxPool2d(torch.nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, input):
        self.input_tensor = input

        self.activations, self.indices = F.max_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, True
        )
        return self.activations

    def simple_grad(self, grad_output):
        grad_input = F.max_unpool2d(
            grad_output.reshape(self.activations.shape),
            self.indices,
            self.kernel_size,
            self.stride,
            self.padding,
            self.input_tensor.shape,
        )
        return grad_input

    def lrp(self, R, lrp_mode="simple"):
        R = self.simple_grad(R)
        return R

