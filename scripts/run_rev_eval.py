"""
Closed-loop evaluation protocols: every controller is run on the same simulated time, long enough
for 1 s decisions.

  acute    PD only, --duration-s (default 40 s), starting from the training start state
           (40 Hz / 0.3 ms / 300 uA, clipped to 250 uA by learned controllers).
  cycling  --blocks blocks of --block-s seconds (default 5 x 60 s), alternating healthy / PD,
           starting healthy.

Controllers (--controller):
  published          the published teacher with its original readout (step-0 decode, state carried
                     over), exactly as in the original experiments
  model:<run_dir>    a teacher or student (train_rev_teacher.py / train_rev_distill.py output) with the
                     revised readout; the window and binning are read from <run_dir>/config.json
  baseline:<run_dir> a non-spiking baseline (train_rev_baseline.py output: MLP/GRU/LSTM)
  baseline_int8:<run_dir> | baseline_int16:<run_dir>   the MLP baseline through the microcontroller
                     package's int8 / int16 fixed-point pipeline
  xylosim:<run_dir>  a distilled student run through bit-accurate XyloSim (needs <run_dir>/xylo/hw_config.json)
  published_ann | published_gru   the original ANN / GRU baseline checkpoints
  constant:F,PW,A    fixed stimulation (Hz, ms, uA)
  unstim | cdbs      no stimulation | 130 Hz / 0.3 ms / 300 uA
  dual_threshold:LO,HI[,F,PW,A]   clinical-style aDBS on the previous window's biomarker: cDBS ON above HI,
                     OFF below LO, hold in between (thresholds are window-specific)
  oracle:F,PW,A      fixed stimulation in PD blocks, off in healthy blocks (true state, no delay)
  integral:TARGET,GAIN,F,PW   amplitude integral law on the previous window's biomarker at fixed
                     frequency and pulse width: amp += GAIN * (beta - TARGET), clipped to [0, 250] uA

Robustness options, applied to the controller's input:
  --pd-scale S       disease severity in PD blocks (the simulator's pd enters the three PD conductance
                     changes continuously; 1 = the trained condition)
  --obs-drop P       each recorded spike is missed with probability P
  --obs-insert R     false spikes added at R Hz per neuron
  --obs-delay K      the controller acts on the observation from K windows earlier

Efficacy must be compared within a window length (beta per window is on a different scale at
100 ms and 1 s), so analysis reports reductions against the unstimulated run of the same
patient at the same window. Charge per decision = freq * amp * pw * window_s.

  python scripts/run_rev_eval.py --controller model:models/revision/teachers_round2/B_p101 \
      --protocol acute --seed 42 --out data/revision/eval
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.environment.gym_pd_rev import count_bin  # noqa: E402
from src.simulation.sim_wrapper import SimulationWrapper  # noqa: E402

TEACHER = os.path.join(ROOT, 'models/final_rockpool_16ch_curriculum/teacher.pth')


class Fixed:
    def __init__(self, f, pw, a):
        self.p = {'freq': f, 'pw': pw, 'amp': a}

    def get_action(self, _):
        return dict(self.p)


class Incremental:
    """Wraps a Q-network that maps (1, T, 80) -> 9 Q-values: per-head argmax increments applied with
    the training environment's leap values and clipping, from the training start state."""

    def __init__(self, net):
        import torch
        self.torch, self.net = torch, net.eval()
        self.freq, self.pw, self.amp = 40.0, 0.3, 300.0

    def get_action(self, spikes):
        with self.torch.no_grad():
            q = self.net(self.torch.as_tensor(np.asarray(spikes), dtype=self.torch.float32).unsqueeze(0))[0]
        inc = [int(self.torch.argmax(q[3 * j:3 * j + 3])) - 1 for j in range(3)]
        self.freq = max(0.0, min(180.0, self.freq + inc[0] * 5.0))
        self.pw = max(0.06, min(0.4, self.pw + inc[1] * 0.1))
        self.amp = max(0.0, min(250.0, self.amp + inc[2] * 5.0))
        return {'freq': self.freq, 'pw': self.pw, 'amp': self.amp}


class DualThreshold:
    def __init__(self, lo, hi, f=130.0, pw=0.3, a=300.0):
        self.lo, self.hi, self.on, self.beta = lo, hi, True, None   # first window: assume high
        self.setting = {'freq': f, 'pw': pw, 'amp': a}               # published: cDBS when on

    def observe(self, beta):
        self.beta = beta

    def get_action(self, _):
        if self.beta is not None:
            if self.beta > self.hi:
                self.on = True
            elif self.beta < self.lo:
                self.on = False
        return dict(self.setting) if self.on else {'freq': 0.0, 'pw': 0.0, 'amp': 0.0}


class Oracle:
    """Knows the true disease state of the upcoming window (no sensing, no delay): the given
    setting in PD blocks, off in healthy blocks. An upper bound on what state-dependent switching
    can save in the cycling protocol (headroom check)."""

    def __init__(self, f, pw, a):
        self.p, self.pd = {'freq': f, 'pw': pw, 'amp': a}, 1

    def set_state(self, pd):
        self.pd = pd

    def get_action(self, _):
        return dict(self.p) if self.pd else {'freq': 0.0, 'pw': 0.0, 'amp': 0.0}


class Integral:
    def __init__(self, target, gain, f, pw):
        self.target, self.gain, self.f, self.pw, self.amp, self.beta = target, gain, f, pw, 250.0, None

    def observe(self, beta):
        self.beta = beta

    def get_action(self, _):
        if self.beta is not None:
            self.amp = max(0.0, min(250.0, self.amp + self.gain * (self.beta - self.target)))
        return {'freq': self.f, 'pw': self.pw, 'amp': self.amp}


def make_controller(spec):
    """Returns (controller, window_ms or None if any, binning, label)."""
    if spec == 'published':
        import quant_fidelity_harness as qh
        return qh.SNNController(TEACHER, 'cpu'), 100, 'hash', 'published'
    if spec.startswith('model:'):
        from src.utils.rev_rl_utils import CorrectedSNNController
        run_dir = spec.split(':', 1)[1]
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        return (CorrectedSNNController(os.path.join(run_dir, 'model.pth'), setting_input=cfg.get('setting_input', False),
                                       out_threshold=cfg.get('out_threshold', 1.0), q_scale=cfg.get('q_scale', 1.0),
                                       linear_bias=cfg.get('linear_bias', True), taus=cfg.get('taus'),
                                       obs_keep=cfg.get('obs_keep')),
                cfg.get('window_ms', 100),
                cfg.get('binning', 'hash'), os.path.basename(run_dir.rstrip('/')))
    if spec.startswith('xylosim:'):
        from src.utils.rev_rl_utils import XyloSimSNNController
        run_dir = spec.split(':', 1)[1]
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        return (XyloSimSNNController(run_dir), cfg.get('window_ms', 100), cfg.get('binning', 'hash'),
                'xylosim_' + os.path.basename(run_dir.rstrip('/')))
    if spec.startswith('baseline_int8:') or spec.startswith('baseline_int16:'):
        # the MLP baseline run through the MCU package's integer pipeline: same decisions a
        # fixed-point microcontroller implementation makes
        bits = 8 if spec.startswith('baseline_int8:') else 16
        from scripts.export_rev_mcu_package import quantise_mlp, mlp_int, round_half_away
        import torch
        from src.models.rev_baselines import build
        run_dir = spec.split(':', 1)[1]
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        assert cfg['arch'] == 'mlp' and cfg['input'] == 'mean80'
        net = build(cfg['arch'], cfg['input'], cfg['hidden'])
        net.load_state_dict(torch.load(os.path.join(run_dir, 'model.pth'), map_location='cpu'))
        P = {k: getattr(net.net[i], a).detach().numpy() for k, i, a in (('W1', 0, 'weight'), ('b1', 0, 'bias'), ('W2', 2, 'weight'), ('b2', 2, 'bias'), ('W3', 4, 'weight'), ('b3', 4, 'bias'))}
        xa = np.load(os.path.join(ROOT, 'data/revision/distill/RT4_p202/windows.npz'))['X'].astype(np.float32).mean(1)
        Q = quantise_mlp(P, xa, bits)
        lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1

        class IntMLP:
            def eval(self):
                return self

            def __call__(self, x):      # x: (1, T, 80) tensor -> (1, 9)
                xq = np.clip(round_half_away(x.numpy().mean(1) / Q['s_x']), lo, hi).astype(np.int64)
                return torch.as_tensor(mlp_int(xq, Q, bits), dtype=torch.float32)
        return (Incremental(IntMLP()), cfg.get('window_ms', 100), cfg.get('binning', 'hash'),
                f'int{bits}_' + os.path.basename(run_dir.rstrip('/')))
    if spec.startswith('baseline:'):
        import torch
        from src.models.rev_baselines import build
        run_dir = spec.split(':', 1)[1]
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        net = build(cfg['arch'], cfg['input'], cfg['hidden'])
        net.load_state_dict(torch.load(os.path.join(run_dir, 'model.pth'), map_location='cpu'))
        return (Incremental(net), cfg.get('window_ms', 100), cfg.get('binning', 'hash'),
                os.path.basename(run_dir.rstrip('/')))
    if spec in ('published_ann', 'published_gru'):
        import torch
        from src.models.ann_baseline import ANNBaseline
        from src.models.rnn_baseline import RNNBaseline
        path, cls = ((os.path.join(ROOT, 'models/ann_baseline_curriculum/ann_baseline.pth'), ANNBaseline)
                     if spec == 'published_ann' else
                     (os.path.join(ROOT, 'models/rnn_baseline_curriculum/rnn_baseline.pth'), RNNBaseline))
        net = cls(80, 128, 9)
        sd = torch.load(path, map_location='cpu')
        net.load_state_dict(sd.get('model_state_dict', sd))
        return Incremental(net), 100, 'hash', spec
    if spec.startswith('constant:'):
        f, pw, a = (float(v) for v in spec.split(':', 1)[1].split(','))
        return Fixed(f, pw, a), None, None, f'constant_{f:g}_{pw:g}_{a:g}'
    if spec.startswith('oracle:'):
        f, pw, a = (float(v) for v in spec.split(':', 1)[1].split(','))
        return Oracle(f, pw, a), None, None, f'oracle_{f:g}_{pw:g}_{a:g}'
    if spec.startswith('dual_threshold:'):
        v = [float(x) for x in spec.split(':', 1)[1].split(',')]
        lo, hi = v[:2]
        if len(v) == 5:
            return DualThreshold(lo, hi, *v[2:]), None, None, f'dual_{lo:g}_{hi:g}_{v[2]:g}_{v[3]:g}_{v[4]:g}'
        return DualThreshold(lo, hi), None, None, f'dual_{lo:g}_{hi:g}'
    if spec.startswith('integral:'):
        tg, gain, f, pw = (float(v) for v in spec.split(':', 1)[1].split(','))
        return Integral(tg, gain, f, pw), None, None, f'integral_{tg:g}_{gain:g}_{f:g}_{pw:g}'
    if spec == 'unstim':
        return Fixed(0.0, 0.0, 0.0), None, None, 'unstim'
    if spec == 'cdbs':
        return Fixed(130.0, 0.3, 300.0), None, None, 'cdbs'
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--controller', required=True)
    ap.add_argument('--protocol', choices=['acute', 'cycling'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--window-ms', type=int, default=None, help='required for non-model controllers')
    ap.add_argument('--duration-s', type=float, default=40.0)
    ap.add_argument('--block-s', type=float, default=60.0)
    ap.add_argument('--blocks', type=int, default=5)
    ap.add_argument('--ctrl-smooth', type=int, default=1, help='simple controllers observe the mean beta of the last N windows (1 = raw window beta)')
    ap.add_argument('--block-jitter', type=float, default=0.0, help='cycling: block lengths drawn uniformly from block-s x [1-j, 1+j], seeded')
    ap.add_argument('--pd-scale', type=float, default=1.0)
    ap.add_argument('--obs-drop', type=float, default=0.0)
    ap.add_argument('--obs-insert', type=float, default=0.0)
    ap.add_argument('--obs-delay', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'data/revision/eval'))
    args = ap.parse_args()

    ctrl, win, binning, label = make_controller(args.controller)
    if win is None:
        if args.window_ms is None:
            sys.exit('--window-ms is required for this controller')
        win, binning = args.window_ms, ('hash' if args.window_ms == 100 else 'count')
    elif args.window_ms not in (None, win):
        sys.exit(f'{label} was trained with {win} ms windows; --window-ms {args.window_ms} conflicts')
    window_s = win / 1000.0

    per_block = int(round(args.block_s / window_s))
    if args.protocol == 'acute':
        schedule = [1] * int(round(args.duration_s / window_s))
    elif args.block_jitter > 0:
        brng = np.random.default_rng(20_000 + args.seed)
        lens = [int(round(args.block_s * brng.uniform(1 - args.block_jitter, 1 + args.block_jitter) / window_s)) for _ in range(args.blocks)]
        schedule = [b % 2 for b, n in enumerate(lens) for _ in range(n)]
    else:
        schedule = [b % 2 for b in range(args.blocks) for _ in range(per_block)]  # healthy first

    def severity(pd):  # set directly: SimulationWrapper.set_pd_state casts to int
        sim.pd_state = float(pd) * args.pd_scale

    rng = np.random.default_rng(10_000 + args.seed)
    sim = SimulationWrapper(n_neurons=10, t_step=win, dt=0.01, seed=args.seed)
    severity(schedule[0])
    obs = sim.reset()
    history = []
    beta_hist = []
    out = {k: [] for k in ('lfp', 'freq', 'pw', 'amp', 'pd_state')}
    for pd in schedule:
        spikes = obs['spike_matrix'] if binning == 'hash' else count_bin(obs['raw_vars'], 100, window_s)
        spikes = np.asarray(spikes, dtype=np.float64)
        if args.obs_drop > 0:
            spikes = rng.binomial(spikes.astype(np.int64), 1.0 - args.obs_drop).astype(np.float64)
        if args.obs_insert > 0:
            spikes = spikes + rng.poisson(args.obs_insert * window_s / 100, spikes.shape)
            if binning == 'hash':
                spikes = np.minimum(spikes, 1.0)
        history.append((spikes, float(obs['lfp'])))
        seen, beta_seen = history[max(0, len(history) - 1 - args.obs_delay)]
        if hasattr(ctrl, 'observe'):
            beta_hist.append(beta_seen)
            ctrl.observe(float(np.mean(beta_hist[-args.ctrl_smooth:])))
        if hasattr(ctrl, 'set_state'):
            ctrl.set_state(pd)
        act = ctrl.get_action(seen)
        severity(pd)
        obs = sim.step(act)
        out['lfp'].append(float(obs['lfp']))
        for k in ('freq', 'pw', 'amp'):
            out[k].append(float(act[k]))
        out['pd_state'].append(pd)

    out.update({'controller': args.controller, 'label': label, 'protocol': args.protocol, 'seed': args.seed,
                'window_ms': win, 'binning': binning, 'duration_s': args.duration_s, 'block_s': args.block_s,
                'blocks': args.blocks, 'block_jitter': args.block_jitter, 'ctrl_smooth': args.ctrl_smooth, 'pd_scale': args.pd_scale, 'obs_drop': args.obs_drop,
                'obs_insert': args.obs_insert, 'obs_delay': args.obs_delay, 'charge': float(sum(f * a * w * window_s for f, a, w in
                                                         zip(out['freq'], out['amp'], out['pw'])))})
    os.makedirs(args.out, exist_ok=True)
    pert = ''.join(f'_{k}{v:g}' for k, v in (('pd', args.pd_scale), ('drop', args.obs_drop),
                                               ('ins', args.obs_insert), ('delay', args.obs_delay))
                   if v != (1.0 if k == 'pd' else 0))
    if args.ctrl_smooth > 1:
        pert += f'_sm{args.ctrl_smooth}'
    if args.protocol == 'cycling' and (args.block_s != 60.0 or args.block_jitter > 0):
        pert += f'_blk{args.block_s:g}' + (f'_jit{args.block_jitter:g}' if args.block_jitter > 0 else '')
    path = os.path.join(args.out, f'{args.protocol}_{label}_w{win}{pert}_seed{args.seed}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f'saved {path} ({len(schedule)} decisions)')


if __name__ == '__main__':
    main()
