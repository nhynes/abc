import torch
from torch import nn


class Bottle(nn.Module):
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
