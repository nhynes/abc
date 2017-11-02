from .gen import GenDataset
from .qa import QADataset

def create(*args, **kwargs):
    return (GenDataset if kwargs.get('synth') else QADataset)(*args, **kwargs)
