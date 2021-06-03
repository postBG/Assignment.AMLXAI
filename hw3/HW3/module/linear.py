import copy

import torch
from torch.nn import functional as F


def s(x):
    return (x > 0).float() * 2 - 1


class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, input):
        self.input_tensor = input
        return F.linear(self.input_tensor, self.weight, self.bias)

    def lrp(self, R, lrp_mode="simple"):
        if lrp_mode == "simple":
            return self._simple_lrp(R)
        elif lrp_mode == "composite":
            return self._composite_lrp(R)
        raise NameError(f"{lrp_mode} is not a valid lrp name")

    def _simple_lrp(self, R, eps=1e-2):
        # dummy answer
        weight = copy.deepcopy(self.weight)
        bias = copy.deepcopy(self.bias)
        x = self.input_tensor

        out = F.linear(x, weight, bias)
        zs = out + s(out) * eps

        R = R / zs
        R_in = F.linear(R, weight.t(), bias=None)
        R_in = R_in * x
        return R_in.detach().cpu()

    def _composite_lrp(self, R):
        """
        Composite lrp use LRP-epsilon for Linear layer and LRP-alpha-beta for Conv layer.
        """
        return self._simple_lrp(R)
