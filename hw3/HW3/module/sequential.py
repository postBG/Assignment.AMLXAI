import torch
import torch.nn.functional as F


class Sequential(torch.nn.Sequential):
    def lrp(self, R, lrp_mode="simple"):
        for module in reversed(self):
            R = module.lrp(R, lrp_mode=lrp_mode)
        return R
