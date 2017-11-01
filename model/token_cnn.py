import torch
from torch import nn
from torch.autograd import Variable


def _construct_layer_specs(max_seq_len):
    kernel_size = 3
    padding_size = 0
    stride = 1
    seq_len = max_seq_len
    layer_specs = []
    while True:
        new_seq_len = (seq_len + 2*padding_size - kernel_size) // stride + 1
        if new_seq_len < 3:
            break
        layer_specs.append(('conv', kernel_size, padding_size, stride))
        seq_len = new_seq_len

        if len(layer_specs) % 4 == 1:
            new_seq_len = (seq_len - 4) // 2 + 1
            if new_seq_len < 3:
                break
            layer_specs.append(('pool', kernel_size, padding_size, stride))
            seq_len = new_seq_len
    assert seq_len > 0
    return layer_specs, seq_len


def _make_seq_emb(layer_specs, tok_emb_dim, seq_emb_dim):
    dim = tok_emb_dim
    dim_step = (seq_emb_dim - dim) // len(layer_specs)
    layers = []
    for i, layer_spec in enumerate(layer_specs, 1):
        function, kernel_size, padding_size, stride = layer_spec
        if function == 'conv':
            out_dim = dim + dim_step
            if i == len(layer_specs):
                out_dim = seq_emb_dim

            layers.append(nn.Conv1d(dim, out_dim, kernel_size,
                                    stride=stride, padding=padding_size))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            dim = out_dim
        elif function == 'pool':
            layers.append(nn.MaxPool1d(4, stride=2))

    return nn.Sequential(*layers)


class TokenCNN(nn.Module):
    """Embeds tokens using a convolutional architecture."""

    def __init__(self, tok_emb, seq_emb_dim,
                 num_toks, max_seq_len, **kwargs):
        super(TokenCNN, self).__init__()

        self.embedding_dim = seq_emb_dim

        self.tok_emb = tok_emb

        layer_specs, final_seq_len = _construct_layer_specs(max_seq_len)
        self.seq_emb = _make_seq_emb(
            layer_specs, self.tok_emb.embedding_dim, seq_emb_dim)

        if kwargs.get('debug'):
            print(self.seq_emb)
            nparam = sum(
                map(lambda p: p.data.numel(), self.seq_emb.parameters()))
            print((f'seq: #layers={len(layer_specs)}, '
                   f'remainder={final_seq_len}, params={nparam}'))

    def forward(self, tokens):
        tok_embs = self.tok_emb(tokens)  # N*max_seq_len*tok_emb_dim
        seq_embs = self.seq_emb(tok_embs.transpose(1, 2))
        return seq_embs.mean(-1)


if __name__ == '__main__':
    opts = {
        'batch_size': 2,
        'tok_emb_dim': 3,
        'seq_emb_dim': 4,
        'num_toks': 5,
        'max_seq_len': 6,
        'debug': True,
    }
    tok_cnn = TokenCNN(**opts)
    embs = tok_cnn(Variable(
        torch.LongTensor(
            opts['batch_size'], opts['max_seq_len']).random_(opts['num_toks'])))
