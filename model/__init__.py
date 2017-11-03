from .generator import RNNGenerator
from .discriminator import Discriminator

def create(*args, **kwargs):
    return SeqGAN(*args, **kwargs)
