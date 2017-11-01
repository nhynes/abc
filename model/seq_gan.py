"""The model."""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as nnf

from generator import Generator
from discriminator import Discriminator

class SeqGAN(nn.Module):
    """An `nn.Module` representing the model."""

    def __init__(self, **kwargs):
        super(SeqGAN, self).__init__()

        self.g_word_emb_dim = kwargs['g_word_emb_dim']
        self.g = Generator(word_emb_dim=self.g_word_emb_dim, **kwargs)
        self.d = Discriminator(word_emb_dim=kwargs['d_word_emb_dim'], **kwargs)


    def forward(self, **unused_kwargs):
        raise NotImplementedError()

    def create_oracle(self, seed):
        """Returns a randomly initialized generator and a random state."""
        rand_state = torch.get_rng_state()
        torch.manual_seed(oracle_seed)

        oracle = Generator(word_emb_dim=self.g_word_emb_dim, **kwargs)
        for param in self.oracle.parameters():
            nn.init.normal(param)

        oracle_rand_state = torch.get_rng_state()
        torch.set_rng_state(rand_state)

        return oracle, oracle_rand_state

    def create_inputs(self):
        """Returns dicts of tensors that this model may use as input."""
        return self.g.create_inputs(), self.d.create_inputs()
        return {
            'inputs': torch.FloatTensor(),
            'outputs_tgt': torch.LongTensor(),
        }


def _test_model():
    debug = True
    model = SeqGAN(**locals())


if __name__ == '__main__':
    _test_model()
