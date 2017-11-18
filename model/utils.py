import torch
from torch import nn


class Apply(nn.Module):
    """A Module that wraps a function."""
    def __init__(self, fn, detach=False):
        super(Apply, self).__init__()
        self.fn = fn
        self.detach = detach

    def forward(self, input):
        output = self.fn(input)
        if self.detach:
            output = output.detach()
        return output


def load_w2v_file(w2v_file):
    """Loads a textual word2vec file in which the tokens are numeric."""
    return torch.np.loadtxt(w2v_file)[:, 1:]
