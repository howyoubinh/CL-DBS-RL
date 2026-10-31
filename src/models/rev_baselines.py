"""
Non-spiking baselines. Every model maps the raw observation
(batch, time, 80) to 9 Q-values (three heads of three), so it plugs into the existing baseline
training loop (scripts/train_rnn_baseline.train with rnn_utils) unchanged.

Input modes
  mean80  time-averaged 80-channel rate vector (the published ANN baseline's input)
  raw80   the full 80-channel sequence (the published GRU baseline's input)
  mean16  the SNN's pooled 16 channels averaged over time; with the MLP this gives the SNN's exact
          layer sizes (16 -> 128 -> 128 -> 9), so spiking vs non-spiking is the only difference
          (the hardware comparison model)
  pool16  the SNN's exact input: the same parameter-free AvgPool1d(5) (80 -> 16 channels) over the
          full sequence, so architecture is the only difference from the SNN

Architectures
  MLPBaseline        two hidden layers; mean80 and mean16 average over time; with pool16 the 16 x T sequence is flattened, so timing
                     information is available to it
  RecurrentBaseline  2-layer GRU or LSTM; Q from the final hidden state; no state carried between calls
"""

import torch
import torch.nn as nn


class Pool16(nn.Module):
    """Identical to the SNN's front end: average groups of 5 neighbouring channels, 80 -> 16."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=5, stride=5)

    def forward(self, x):
        B, T, C = x.shape
        return self.pool(x.reshape(B * T, 1, C)).reshape(B, T, -1)


class MLPBaseline(nn.Module):
    def __init__(self, input_mode='mean80', hidden=128, n_actions=9, num_steps=100):
        super().__init__()
        assert input_mode in ('mean80', 'mean16', 'pool16')
        self.input_mode = input_mode
        self.pool = Pool16() if input_mode in ('mean16', 'pool16') else None
        n_in = {'mean80': 80, 'mean16': 16, 'pool16': 16 * num_steps}[input_mode]
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        if self.input_mode == 'mean80':
            x = x.mean(dim=1)
        elif self.input_mode == 'mean16':
            x = self.pool(x).mean(dim=1)
        else:
            x = self.pool(x).flatten(1)
        return self.net(x)


class RecurrentBaseline(nn.Module):
    def __init__(self, cell='gru', input_mode='raw80', hidden=128, n_actions=9, num_layers=2):
        super().__init__()
        assert cell in ('gru', 'lstm') and input_mode in ('raw80', 'pool16')
        self.pool = Pool16() if input_mode == 'pool16' else None
        rnn = nn.GRU if cell == 'gru' else nn.LSTM
        self.rnn = rnn(input_size=80 if input_mode == 'raw80' else 16, hidden_size=hidden,
                       num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, n_actions)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


def build(arch, input_mode, hidden, num_steps=100):
    if arch == 'mlp':
        return MLPBaseline(input_mode, hidden, num_steps=num_steps)
    return RecurrentBaseline(arch, input_mode, hidden)
