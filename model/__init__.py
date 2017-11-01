# from . import SeqGAN
from .generator import Generator
from .discriminator import Discriminator

def create(*args, **kwargs):
    return SeqGAN(*args, **kwargs)
