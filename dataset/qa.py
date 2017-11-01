"""A Dataset that loads the Amazon product QA."""

import torch
import torch.utils.data

import common


class QADataset(torch.utils.data.Dataset):
    """Loads the data."""

    def __init__(self, dataset, part, **unused_kwargs):
        super(QADataset, self).__init__()

        self.part = part

        self.samples = common.unpickle(f'{dataset}/{part}.pkl')

    def __getitem__(self, index):
        inputs, outputs = self.samples[index]

        return {
            'inputs': inputs,
            'outputs_tgt': outputs,
        }

    def __len__(self):
        return len(self.samples)


def create(*args, **kwargs):
    """Returns a QADataset."""
    return QADataset(*args, **kwargs)


def test_dataset():
    dataset = 'data/dataset'
    part = 'test'
    debug = True

    ds_test = QADataset(**locals())
    datum = ds_test[0]

    for i in torch.randperm(len(ds_test))[:1000]:
        datum = ds_test[i]
