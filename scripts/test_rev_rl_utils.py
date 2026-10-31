"""
Tests for src/utils/rev_rl_utils.py (per-sample Q with state reset, greedy acting, evaluation
controller, optimisation step), plus a measurement of how the initial-submission pipeline's
Q-values and decoded actions differ from the revised ones on real simulator windows, with the
published teacher's weights.

  python scripts/test_rev_rl_utils.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from scripts.train_rev_teacher import AvgPoolDownsampler  # noqa: E402
from src.environment.replay_memory import ReplayMemory  # noqa: E402
from src.models.rockpool_dqsn import RockpoolDQSN  # noqa: E402
from src.simulation.sim_wrapper import SimulationWrapper  # noqa: E402
from src.utils import rev_rl_utils as R  # noqa: E402
from src.utils.action_utils_rockpool import get_action_dict  # noqa: E402

TEACHER = os.path.join(ROOT, 'models/final_rockpool_16ch_curriculum/teacher.pth')
dev = torch.device('cpu')


def net_from(path=None):
    n = nn.Sequential(AvgPoolDownsampler(5, 5), RockpoolDQSN(16, 128, 0.95, 9, 100, 1, use_mempot=True, dt=10e-3))
    if path:
        n.load_state_dict(torch.load(path, map_location='cpu'))
    return n


def real_windows(n_each=12):
    """Real 100 ms observation windows (100 x 80 binary) from the simulator, PD and healthy."""
    out = []
    for pd in (1, 0):
        sim = SimulationWrapper(n_neurons=10, t_step=100, dt=0.01, seed=42)
        sim.set_pd_state(pd)
        obs = sim.reset()
        for i in range(n_each):
            obs = sim.step({'freq': 180.0, 'pw': 0.06, 'amp': 212.0} if i % 2 else {'freq': 0, 'pw': 0, 'amp': 0})
            out.append(obs['spike_matrix'].astype(np.float32))
    return torch.tensor(np.stack(out))


def main():
    torch.manual_seed(0)
    teacher = net_from(TEACHER)
    X = real_windows()                                   # (24, 100, 80)
    print(f'real windows: {tuple(X.shape)}')

    # 1. per-sample Q does not depend on batch composition or on earlier calls
    q_batch, _, _ = R.q_values(teacher, X)
    q_single = torch.cat([R.q_values(teacher, X[i:i + 1])[0] for i in range(len(X))])
    rel = float((q_batch - q_single).abs().max() / q_single.abs().mean())
    assert rel < 1e-5, rel
    R.q_values(teacher, X.flip(0))                        # unrelated call in between
    q_again = R.q_values(teacher, X[:1])[0]
    assert torch.allclose(q_again, q_single[:1], rtol=1e-5, atol=1e-3)
    print(f'[pass] Q is per-sample and history-independent (max rel diff {rel:.1e})')

    # 2. greedy acting == argmax of the training Q, regardless of what ran before
    class E:  # minimal env stub (random branch never taken with eps = 0)
        action_space = None
    for i in range(len(X)):
        R.q_values(teacher, X[(i + 7) % len(X):][:1])     # disturb state
        a, _ = R.select_action_rev(teacher, teacher, X[i:i + 1], E, 10**9, dev, 0.0, 0.0, 1.0)
        assert a == R.decode(q_single[i]), (i, a)
    print('[pass] greedy select_action_rev == decode(training Q) on every window')

    # 3. evaluation controller == acting
    ctrl = R.CorrectedSNNController(TEACHER)
    assert all(ctrl.increments(X[i].numpy()) == R.decode(q_single[i]) for i in range(len(X)))
    print('[pass] CorrectedSNNController increments == greedy action')

    # 4. optimization step runs and fits a fixed batch
    pol, tgt = net_from(), net_from()
    tgt.load_state_dict(pol.state_dict())
    opt = optim.AdamW(pol.parameters(), lr=1e-3, amsgrad=True)
    mem = ReplayMemory(1000)
    rng = np.random.default_rng(0)
    for i in range(32):
        a = {h: int(rng.integers(-1, 2)) for h in R.HEADS}
        mem.push(X[i % len(X)].unsqueeze(0), a, X[(i + 1) % len(X)].unsqueeze(0), torch.tensor([float(rng.normal())]))
    losses = [R.optimize_model_rev(pol, tgt, opt, mem, dev, 32, 0.0)[0] for _ in range(40)]
    assert losses[-1] < losses[0], losses[::10]
    print(f'[pass] optimize_model_rev fits a fixed batch (gamma=0): loss {losses[0]:.3g} -> {losses[-1]:.3g}')

    # 5. the difference from the original pipeline (informational), with resets for the reference
    #    (a) training Q: original permute to (T, B, C), state carried over
    _, mem_tr, *_ = teacher(X.permute(1, 0, 2))
    q_orig_train = mem_tr.sum(0)
    err = (q_orig_train - q_single).abs() / q_single.abs().clamp_min(1e-6)
    print(f'[info] original training Q vs revised Q: median rel difference {float(err.median()):.1%}, '
          f'max {float(err.max()):.1%}')
    #    (b) acting: original decode = element [0] of get_action_dict, state carried over
    agree, n = 0, 0
    for i in range(len(X)):
        _, m, *_ = teacher(X[i:i + 1])                    # no reset: state carries over, as in the original
        a_orig = get_action_dict(teacher, None, m)[0]
        agree += (a_orig == R.decode(q_single[i])); n += 1
    print(f'[info] original acting decode agrees with the revised readout on {agree}/{n} windows (joint action)')
    print('all tests passed')


if __name__ == '__main__':
    main()
