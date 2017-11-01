from .synthetic import SynthDataset
from .qa import QADataset

def create(*args, **kwargs):
    return (SynthDataset if kwargs.get('synth') else QADataset)(*args, **kwargs)
