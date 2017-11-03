# from . import SeqGAN
from .generator import RNNGenerator, CNNGenerator
from .discriminator import Discriminator

def create(*args, **kwargs):
    return SeqGAN(*args, **kwargs)
