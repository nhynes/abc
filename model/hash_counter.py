"""A Module that uses locality sensitve hashing to count visited states."""

import torch
from torch import nn
from torch.autograd import Variable


class HashCounter(nn.Module):
    """Accumulates counts of items that hash to a particular value."""

    def __init__(self, hash_fn, num_hash_buckets, **unused_kwargs):
        super(HashCounter, self).__init__()

        self.code_len = int(torch.np.ceil(torch.np.log2(num_hash_buckets)))
        self.hash_fn = hash_fn

        self.register_buffer('counts', torch.zeros(num_hash_buckets))

        self.register_buffer('_ones', torch.ones(1))
        self.register_buffer('_powers_of_two',
                             2**torch.arange(self.code_len-1, -1, -1))

    def forward(self, items, accumulator='counts', **unused_kwargs):
        """ Accumulates hashed item counts.

        items: N*(size of single item accepted by hash_fn)
        accumulator: the name of an accumulator

        Returns: A LongTensor of hash_bucket indices: N
        """
        ones = self._ones.expand(items.size(0))
        acc = self._buffers.get(accumulator, torch.zeros_like(self.counts))
        if accumulator not in self._buffers:
            self.register_buffer(accumulator, acc)

        hash_codes = self.hash_fn(items).data  # N*code_len
        hash_buckets = (hash_codes @ self._powers_of_two).long()  # N
        acc.put_(hash_buckets, ones, accumulate=True)

        return hash_buckets


def create(hash_fn, **opts):
    """Creates a token generator."""
    return HashCounter(hash_fn, **opts)


def test_simhash_table():
    """Tests the HashCounter."""
    # pylint: disable=too-many-locals,unused-variable

    num_hash_buckets = 4
    debug = True

    class HashFn(object):
        """A mock hash function. Big-endian."""
        codes = None
        buckets = None

        @staticmethod
        def _i2b(i):
            bitwidth = int(torch.np.log2(num_hash_buckets))
            bin_rep = list(map(int, bin(i)[2:]))
            return [0]*(bitwidth - len(bin_rep)) + bin_rep

        def set_codes(self, bin_counts):
            """Sets the big-endian binary codes that the HashFn will return."""
            codes = []
            buckets = []
            for i, count in enumerate(bin_counts):
                codes.extend([self._i2b(i)]*count)
                buckets.extend([i]*count)
            rp = torch.randperm(len(codes))
            self.codes = torch.FloatTensor(codes)[rp]
            self.buckets = torch.LongTensor(buckets)[rp]

        def __call__(self, _):
            return Variable(self.codes)

    hash_fn = HashFn()
    simhash_table = HashCounter(**locals())

    expected_counts_train = [1, 2, 0, 4]
    hash_fn.set_codes(expected_counts_train)
    toks = Variable(torch.LongTensor(sum(expected_counts_train), 4))

    assert (simhash_table(toks, 'counts2') == hash_fn.buckets).all()
    assert (simhash_table.counts2.numpy() == expected_counts_train).all()
    assert (simhash_table.counts == 0).all()

    expected_counts_test = [4, 3, 2, 1]
    hash_fn.set_codes(expected_counts_test)
    toks = Variable(torch.LongTensor(sum(expected_counts_test), 4))

    assert (simhash_table(toks) == hash_fn.buckets).all()
    assert (simhash_table.counts2.numpy() == expected_counts_train).all()
    assert (simhash_table.counts.numpy() == expected_counts_test).all()
