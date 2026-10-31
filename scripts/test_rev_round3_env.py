"""
Tests for src/environment/gym_pd_round3.py and the matching network/controller options.

  1. energy: the simulator's pulse-train RMS matches the closed form A * sqrt(f * pw / 1000) used
     in the offline reward check, for f >= 40 Hz; E_max ~ 67 uA. (creatdbs restarts the pulse train
     at t = 0 in every 100 ms window, so a window holds ceil(0.1 f) pulses: at 5 Hz that is one
     pulse per window, twice the nominal charge. Above 40 Hz the rounding is < 6%, and at the
     180 Hz operating points it is +2.7% in charge. Reported separately; the env uses the
     simulator's actual RMS, so training sees the true cost.)
  2. reward replay: the env's calc_reward on a recorded cycling run (oracle 250 uA, patient 42)
     reproduces scripts/analyze_rev_reward_candidates.py's roll10 / rms67 / tau 290 reward
     window for window.
  3. integration (real simulator, patient 101, 2 short episodes): transitions flip the state at
     switch_at and only there; random_start draws settings on the action grid; the reward each
     step equals a recomputation from the logged beta history and E; reset clears the history.
  4. setting-input channels (83-channel observation), non-spiking Q readout, q_scale, and
     identical observations built by the env and by CorrectedSNNController.

  python scripts/test_rev_round3_env.py
"""

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import analyze_rev_reward_candidates as RC  # noqa: E402
from src.environment.gym_pd_round3 import MousePDEnvRound3, max_rms_current  # noqa: E402
from src.simulation.simulate_network_optimized import creatdbs  # noqa: E402

ENV_KW = dict(leap={'pw': 0.1, 'amp': 5.0, 'freq': 5.0}, num_steps=100, tau_beta_max=290.0, tau_reward=3000.,
              delta=1.0, TMAX=100, thresh_time_req=999, binning='hash')


def test_energy():
    worst, low = 0.0, {}
    for f, pw, a in itertools.product((5, 10, 20, 40, 130, 180), (0.06, 0.1, 0.3, 0.4), (50, 212, 250)):
        rms = np.sqrt(np.mean(np.square(creatdbs(f, 100, 0.01, pw, a))))
        closed = a * np.sqrt(f * pw / 1000)
        if f >= 40:
            worst = max(worst, abs(rms - closed) / closed)
        else:
            low[f] = rms / closed
    e_max = max_rms_current(100, 0.01)
    print(f'1. energy: f >= 40 Hz: max |sim RMS - closed form| / closed form = {100 * worst:.2f}%; '
          f'below 40 Hz the window-restart rounding gives sim/closed = '
          + ', '.join(f'{f} Hz {r:.2f}' for f, r in low.items()) + f'; E_max = {e_max:.2f} uA')
    assert worst < 0.06, worst
    assert abs(e_max - RC.E_RMS_MAX) / RC.E_RMS_MAX < 0.03, (e_max, RC.E_RMS_MAX)


def test_reward_replay():
    p = os.path.join(ROOT, 'data/revision/headroom/cycling_oracle_180_0.06_250_w100_seed42.json')
    d = json.load(open(p))
    beta = np.asarray(d['lfp'], float)
    stim = np.stack([np.asarray(d[k], float) for k in ('freq', 'pw', 'amp')], 1)
    ref = RC.reward(RC.roll(beta, 10), RC.energy(stim, 'rms67'), 290.0, 30.0)

    env = MousePDEnvRound3.__new__(MousePDEnvRound3)      # reward logic only; no simulator
    env.beta_smooth_windows, env.tau_beta_max, env.tau_reward, env.delta = 10, 290.0, 3000., 1.0
    env.E_max = RC.E_RMS_MAX
    env.reward_scale = 1.0
    env._beta_hist = []
    got = []
    for b, (f, pw, a) in zip(beta, stim):
        env._beta_hist.append(float(b))
        env.beta_smooth = float(np.mean(env._beta_hist[-10:]))
        env.E = a * np.sqrt(f * pw / 1000)
        env.calc_reward()
        got.append(env.reward)
    err = np.abs(np.asarray(got) - ref).max()
    print(f'2. reward replay on {os.path.basename(p)}: {len(got)} windows, max |env - offline| = {err:.2e}, '
          f'mean reward {np.mean(got):.1f}')
    assert err < 1e-6, err


def test_integration():
    env = MousePDEnvRound3(max_steps=30, seed=101, transitions=True, switch_min=5, random_start=True,
                           rng_seed=7, **ENV_KW)
    e_max = env.E_max
    for ep in range(2):
        env.set_pd_state(ep % 2)
        env.reset()
        assert env._beta_hist and len(env._beta_hist) == 1, 'reset must clear the history (priming window only)'
        assert env.switch_at is not None and 5 <= env.switch_at <= 25, env.switch_at
        assert env.freq in {5.0 * k for k in range(37)} and 0.0 <= env.amp <= 250.0 and env.amp % 5 == 0
        assert env.pw in (0.06, 0.1, 0.2, 0.3, 0.4), env.pw
        start, hist, rewards = (env.freq, env.pw, env.amp), [env._beta_hist[0]], []
        pd0 = env.PD
        for t in range(1, 31):
            _, r, term, trunc, _ = env.step({'freq': 0, 'pw': 0, 'amp': 0})
            hist.append(float(env.gpi_alpha_beta_area))
            want_pd = pd0 if t < env.switch_at else 1 - pd0
            assert env.PD == want_pd, (t, env.PD, want_pd, env.switch_at)
            bs = np.mean(hist[-10:])
            e = min(env.E / e_max, 1.0)
            want = -30.0 * (bs - 290.0) if bs > 290.0 else 3000.0 * (1.0 - e)
            assert abs(r - want) < 1e-6, (t, r, want)
            rewards.append(r)
        print(f'3. episode {ep}: start state {pd0}, switch at step {env.switch_at}, start setting '
              f'{start[0]:g} Hz / {start[1]:g} ms / {start[2]:g} uA, pd_trace flips once: '
              f'{sum(np.diff(env.pd_trace) != 0) == 1}, mean reward {np.mean(rewards):.0f}')
        assert sum(np.diff(env.pd_trace) != 0) == 1


def test_round4():
    """Setting channels, 83-channel observation, non-spiking readout, controller/env consistency."""
    import torch
    from src.environment.gym_pd_round3 import setting_channels
    from src.utils.rev_rl_utils import CorrectedSNNController, q_values
    from scripts.train_rev_teacher import make_net
    ch = setting_channels((90.0, 0.2, 125.0), 100)
    assert ch.shape == (100, 3) and ch.sum(0).tolist() == [50, 50, 50], ch.sum(0)
    assert setting_channels((0, 0, 0), 100).sum() == 0 and setting_channels((180, 0.4, 300), 100).sum(0).tolist() == [100, 100, 100]
    gaps = np.diff(np.nonzero(ch[:, 2])[0])
    assert gaps.min() == gaps.max() == 2, 'spikes must be evenly spaced'

    env = MousePDEnvRound3(max_steps=4, seed=101, setting_input=True, reward_scale=3000.0, rng_seed=3, **ENV_KW)
    env.set_pd_state(1)
    _, info = env.reset()
    obs0 = info['raw_spike_data']
    assert obs0.shape == (100, 83) and obs0[:, 80:].sum() == 0, 'priming window is unstimulated -> zero channels'
    _, r, _, _, info = env.step({'freq': 1, 'pw': 0, 'amp': -1})     # 45 Hz / 0.3 ms / 250 uA (300 clipped)
    obs1 = info['raw_spike_data']
    assert np.array_equal(obs1[:, 80:], setting_channels((45.0, 0.3, 250.0), 100))
    assert -1.0 <= r <= 1.0, r

    net = make_net(100, setting_input=True, out_threshold=1e6)
    assert net[1].out_threshold == 1e6
    x = torch.as_tensor(np.stack([obs0, obs1]), dtype=torch.float32)
    q, _, _ = q_values(net, x)
    assert q.shape == (2, 9)
    reset_q = q_values(net, x[:1])[0]
    assert torch.allclose(reset_q, q[:1], atol=1e-4)
    # non-spiking readout: no output spikes, membrane unbounded above 1
    from src.utils.rev_rl_utils import reset_state
    reset_state(net)
    with torch.no_grad():
        spk, mem, _, _, _ = net(x * 50)   # strong drive
    assert spk.sum() == 0 and mem.max() > 1.0, (spk.sum().item(), mem.max().item())

    # q_scale divides the readout; the controller must apply the same scale
    from scripts.train_rev_teacher import calibrate_q_scale
    qs = calibrate_q_scale(net, True, 101, 'cpu')
    net[1].q_scale = qs
    q_scaled = q_values(net, x)[0]
    assert torch.allclose(q_scaled * qs, q, atol=1e-2 * qs), 'q_scale must divide the summed readout'
    q = q_scaled
    # controller appends the same channels the env would, given the same applied setting
    torch.save(net.state_dict(), '/tmp/_rev_round4_test.pth')
    ctrl = CorrectedSNNController('/tmp/_rev_round4_test.pth', setting_input=True, out_threshold=1e6, q_scale=qs)
    inc0 = ctrl.increments(obs0[:, :80])             # first window: controller assumes unstimulated
    ctrl.get_action(obs0[:, :80])                     # applies its increments from 40/0.3/300
    ctrl.freq, ctrl.pw, ctrl.amp = 45.0, 0.3, 250.0
    ctrl.applied = (45.0, 0.3, 250.0)
    q_ctrl = q_values(ctrl.net, torch.as_tensor(np.concatenate([obs1[:, :80], setting_channels(ctrl.applied, 100)], 1),
                                                dtype=torch.float32).unsqueeze(0))[0]
    q_env = q_values(net, x[1:2])[0]
    assert torch.allclose(q_ctrl, q_env, atol=1e-4), 'controller and env must build identical observations'
    print(f'4. round-4: setting channels ok; env obs {obs0.shape}; reward scaled {r:.3f}; non-spiking readout '
          f'Q range on strong drive {mem.sum(1).min().item():.1f}..{mem.sum(1).max().item():.1f}; '
          f'auto q_scale {qs:.0f} -> |Q| on real windows {q.abs().mean().item():.2f}; controller == env')


if __name__ == '__main__':
    test_energy()
    test_reward_replay()
    test_integration()
    test_round4()
    print('all round-3/4 env tests passed')
