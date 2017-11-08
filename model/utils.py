import torch
from torch import nn
import numpy as np


class DontTrain(nn.Module):
    def __init__(self, mod):
        super(DontTrain, self).__init__()
        self.mod = mod

    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs).detach()


def load_w2v_file(w2v_file):
    return np.loadtxt(w2v_file)[:, 1:]
