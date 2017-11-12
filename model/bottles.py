"""Nd wrappers for modules that operate on the columns of a matrix."""

import torch
from torch import nn


class Bottle(nn.Module):
    """Allows a 2D module to process an Nd input."""

    def forward(self, inp):
        sz = inp.size()
        if len(sz) <= 2:
            return super(Bottle, self).forward(inp)
        out = super(Bottle, self).forward(inp.view(-1, sz[-1]))
        out_sz = out.size()
        return out.view(*sz[:-1], *out_sz[-(len(out_sz) - 1):])


class BottledLinear(Bottle, nn.Linear):
    pass


class BottledEmbedding(Bottle, nn.Embedding):
    pass
