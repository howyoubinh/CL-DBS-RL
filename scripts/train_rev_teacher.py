"""
SNN teacher training (spiking DQN) with configurable reward, patient, window and environment.

Same training as the published teacher (W&B 3hvigb6o):
    python -m scripts.train_rl_rockpool_16ch --num-episodes 500 --curriculum --delta 1.0 \
        --healthy-duration 100 --max-steps 50 --pd-duration 100
The defaults below reproduce that command. It reuses train() from
scripts/train_rl_rockpool_16ch.py unchanged, with the same network, hyperparameters and
curriculum. Differences, all deliberate:

  --tau-beta      reward threshold (published: hardcoded 150)
  --patient-seed  the simulated patient (published: random.randint at first reset, never logged)
  --agent-seed    seeds Python/torch RNGs for exploration and replay (published: unseeded)
  --pipeline      'original' = the published select_action/optimize_model; 'corrected' = per-sample
                  Q with state reset (src/utils/rev_rl_utils.py) swapped into the same train()
  --window-ms     observation/decision window (published: 100); --binning count for windows != 100
  environment     MousePDEnvAdaptiveRev: a failed spectral estimate counts as missing, not beta = 0
  --env round3    MousePDEnvRound3: reward on beta averaged over
                  --beta-smooth windows, energy normalised by --energy-norm ('max' = the action
                  space's maximum RMS, or a number such as 250); --transitions switches the state
                  once per episode at a random step; --random-start starts episodes from a random
                  setting (off with probability --p-off). Arm R = round3 without the last two;
                  arm RT = round3 with both (and --max-steps 100).
  outputs         everything for a run goes in its own folder (train() writes checkpoints to
                  relative paths, so each run executes from its folder), including a copy of
                  the source files it ran with

Usage:
  python scripts/train_rev_teacher.py --tau-beta 290 --patient-seed 101 --run-name tau290_p101
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import wandb  # noqa: E402

import scripts.train_rl_rockpool_16ch as base  # noqa: E402  (train() + hyperparameters)
from src.environment.gym_pd_rev import MousePDEnvAdaptiveRev  # noqa: E402
from src.environment.gym_pd_round3 import MousePDEnvRound3  # noqa: E402
from src.environment.replay_memory import ReplayMemory  # noqa: E402
from src.models.rockpool_dqsn import RockpoolDQSN  # noqa: E402
from src.utils.io_utils import save_results  # noqa: E402
from src.utils import rev_rl_utils  # noqa: E402

SNAPSHOT = [
    'scripts/train_rev_teacher.py', 'scripts/train_rl_rockpool_16ch.py',
    'src/environment/gym_pd.py', 'src/environment/gym_pd_rev.py', 'src/environment/gym_pd_round3.py',
    'src/simulation/simulate_network_optimized.py', 'src/simulation/sim_wrapper.py',
    'src/models/rockpool_dqsn.py', 'src/utils/action_utils_rockpool.py', 'src/utils/rev_rl_utils.py',
    'src/utils/optimization_utils_rockpool.py', 'src/utils/data_processing.py',
]


class AvgPoolDownsampler(nn.Module):
    """Identical to the class defined inside train_rl_rockpool_16ch.main(): 80 -> 16 channels."""

    def __init__(self, kernel_size=5, stride=5):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        B, T, C = x.shape
        return self.pool(x.reshape(B * T, 1, C)).reshape(B, T, -1)


class AvgPoolKeepTail(nn.Module):
    """80 -> 16 channels by average pooling, with any channels beyond the first 80 (the stimulation
    setting channels of --setting-input) passed through unpooled: (B, T, 80 + k) -> (B, T, 16 + k)."""

    def __init__(self, n_pooled=80, kernel_size=5, stride=5):
        super().__init__()
        self.n_pooled = n_pooled
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        B, T, C = x.shape
        head = self.pool(x[..., :self.n_pooled].reshape(B * T, 1, self.n_pooled)).reshape(B, T, -1)
        return torch.cat([head, x[..., self.n_pooled:]], dim=2) if C > self.n_pooled else head


XYLO_TAUS = {'mem': -10e-3 / __import__('math').log(1 - 2 ** -3), 'syn': -10e-3 / __import__('math').log(1 - 2 ** -2)}  # 74.9 ms, 34.8 ms


def make_net(num_steps, setting_input=False, out_threshold=1.0, device='cpu', linear_bias=True, taus=None):
    """taus: None = published (100 ms, 50 ms); 'xylo' = the time constants whose exp(-dt/tau) equals the
    Xylo bit-shift decays 1 - 2^-3 (membrane) and 1 - 2^-2 (synapse) that the mapper assigns to the
    published values, so the chip's dynamics match the trained model."""
    tm, ts = (XYLO_TAUS['mem'], XYLO_TAUS['syn']) if taus == 'xylo' else (100e-3, 50e-3)
    n_in = 16 + (3 if setting_input else 0)
    rock = RockpoolDQSN(n_in, 128, base.BETA, 9, num_steps, base.BATCH_SIZE, use_mempot=True, dt=10e-3,
                        target_sparsity=0.0, sparsity_weight=0.0, out_threshold=out_threshold, linear_bias=linear_bias,
                        tau_mem=tm, tau_syn=ts)
    pool = AvgPoolKeepTail(80, 5, 5) if setting_input else AvgPoolDownsampler(5, 5)
    return nn.Sequential(pool, rock).to(device)


def calibrate_q_scale(net, setting_input, patient_seed, device):
    """Std of the untrained net's sum_t Vmem over recorded 100 ms windows of the training patient
    (data/revision/obs: unstimulated and 212 uA, PD and healthy), with the setting channels the env
    would append. Used as q_scale so Q starts at O(1); stored in config.json for evaluation."""
    from src.environment.gym_pd_round3 import setting_channels
    X = []
    for cond, setting in (('unstim', (0.0, 0.0, 0.0)), ('c212', (180.0, 0.06, 212.0))):
        for state in ('healthy', 'pd'):
            p = os.path.join(ROOT, f'data/revision/obs/obs_{cond}_{state}_seed{patient_seed}.npz')
            w = np.minimum(np.load(p)['counts'][10:74].astype(np.float32), 1.0)
            if setting_input:
                w = np.concatenate([w, np.broadcast_to(setting_channels(setting, w.shape[1]), (len(w),) + (w.shape[1], 3))], 2)
            X.append(w)
    X = torch.as_tensor(np.concatenate(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        rev_rl_utils.reset_state(net)
        _, mem, _, _, _ = net(X)
    return float(mem.sum(1).std())


def build_net(args, num_steps):
    net = make_net(num_steps, args.setting_input, args.out_threshold, base.device)
    if args.q_scale == 'auto':
        net[1].q_scale = calibrate_q_scale(net, args.setting_input, args.patient_seed, base.device)
    else:
        net[1].q_scale = float(args.q_scale)
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tau-beta', type=float, default=150.0)
    ap.add_argument('--patient-seed', type=int, required=True)
    ap.add_argument('--agent-seed', type=int, default=None, help='default: same as --patient-seed')
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--out-root', default=os.path.join(ROOT, 'models/revision/teachers'))
    ap.add_argument('--delta', type=float, default=1.0)
    ap.add_argument('--num-episodes', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=50)
    ap.add_argument('--pd-duration', type=int, default=100)
    ap.add_argument('--healthy-duration', type=int, default=100)
    ap.add_argument('--no-curriculum', action='store_true')
    ap.add_argument('--pipeline', choices=['original', 'corrected'], required=True)
    ap.add_argument('--window-ms', type=int, default=100)
    ap.add_argument('--binning', choices=['hash', 'count'], default='hash')
    ap.add_argument('--env', choices=['rev', 'round3'], default='rev')
    ap.add_argument('--beta-smooth', type=int, default=10, help='round3: windows averaged in the reward')
    ap.add_argument('--energy-norm', default='max', help="round3: 'max' or a number (published: 250)")
    ap.add_argument('--transitions', action='store_true', help='round3: one state switch per episode')
    ap.add_argument('--switch-min', type=int, default=10, help='round3: earliest/latest switch margin (steps)')
    ap.add_argument('--random-start', action='store_true', help='round3: random start setting per episode')
    ap.add_argument('--p-off', type=float, default=0.25, help='round3: probability the random start is off')
    ap.add_argument('--obs-keep', type=lambda v: v.split(','), default=None, help='round3: keep only these populations (th,stn,gpe,gpi,strd2,strd1,cor); others zeroed')
    ap.add_argument('--obs-noise', type=float, default=0.0, help='round3: Poisson background spikes added in training, Hz per neuron')
    ap.add_argument('--obs-drop', type=float, default=0.0, help='round3: fraction of recorded spikes dropped in training')
    ap.add_argument('--setting-input', action='store_true',
                    help='round3: 3 extra observation channels carrying the current stimulation setting')
    ap.add_argument('--out-threshold', type=float, default=1.0,
                    help='output LIF threshold (1e6 = non-spiking membrane readout)')
    ap.add_argument('--reward-scale', type=float, default=1.0, help='round3: rewards are divided by this')
    ap.add_argument('--q-scale', default='1.0',
                    help="Q = sum_t Vmem / q_scale; 'auto' = std of the untrained readout on recorded windows")
    args = ap.parse_args()
    if (args.setting_input or args.reward_scale != 1.0) and args.env != 'round3':
        sys.exit('--setting-input / --reward-scale need --env round3')
    agent_seed = args.patient_seed if args.agent_seed is None else args.agent_seed
    if args.window_ms != 100 and args.binning != 'count':
        sys.exit('windows other than 100 ms need --binning count (constructor_hash assumes 0-0.1 s)')
    if args.pipeline == 'corrected':  # train() looks these up as module globals
        base.select_action = rev_rl_utils.select_action_rev
        base.optimize_model = rev_rl_utils.optimize_model_rev

    run_dir = os.path.join(args.out_root, args.run_name)
    if os.path.exists(os.path.join(run_dir, 'model.pth')):
        sys.exit(f'{run_dir}/model.pth already exists; refusing to overwrite')
    os.makedirs(os.path.join(run_dir, 'code'), exist_ok=True)
    for rel in SNAPSHOT:
        dst = os.path.join(run_dir, 'code', rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)

    random.seed(agent_seed)
    torch.manual_seed(agent_seed)

    env_kw = dict(leap={'pw': 0.1, 'amp': 5.0, 'freq': 5.0}, num_steps=100, tau_beta_max=args.tau_beta,
                  tau_reward=3000., delta=args.delta, max_steps=args.max_steps, TMAX=args.window_ms,
                  thresh_time_req=999, seed=args.patient_seed, binning=args.binning)
    if args.env == 'round3':
        env = MousePDEnvRound3(beta_smooth_windows=args.beta_smooth, energy_norm=args.energy_norm,
                               transitions=args.transitions, switch_min=args.switch_min,
                               random_start=args.random_start, p_off=args.p_off, rng_seed=agent_seed,
                               setting_input=args.setting_input, reward_scale=args.reward_scale,
                               obs_keep=args.obs_keep, obs_noise_hz=args.obs_noise, obs_drop=args.obs_drop, **env_kw)
    else:
        env = MousePDEnvAdaptiveRev(**env_kw)

    policy_net, target_net = build_net(args, env.num_steps), build_net(args, env.num_steps)
    target_net.load_state_dict(policy_net.state_dict())
    target_net[1].q_scale = policy_net[1].q_scale
    print(f'q_scale = {policy_net[1].q_scale:.1f}')
    optimizer = optim.AdamW(policy_net.parameters(), lr=base.LR, amsgrad=True)
    memory = ReplayMemory(100000)

    try:
        commit = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = None
    config = {
        'run_name': args.run_name, 'argv': sys.argv, 'started': datetime.now().isoformat(timespec='seconds'),
        'git_commit_public_repo': commit, 'note': 'working tree may differ from commit; see code/ snapshot',
        'pipeline': args.pipeline, 'window_ms': args.window_ms, 'binning': args.binning,
        'tau_beta_max': args.tau_beta, 'patient_seed': args.patient_seed, 'agent_seed': agent_seed,
        'delta': args.delta, 'num_episodes': args.num_episodes, 'max_steps': args.max_steps,
        'curriculum': not args.no_curriculum, 'pd_duration': args.pd_duration,
        'healthy_duration': args.healthy_duration, 'env_class': env.__class__.__name__,
        'env': args.env, 'beta_smooth_windows': getattr(env, 'beta_smooth_windows', 1),
        'energy_norm': args.energy_norm if args.env == 'round3' else '250',
        'E_max': getattr(env, 'E_max', 250.0), 'transitions': args.transitions, 'obs_keep': args.obs_keep, 'obs_noise': args.obs_noise, 'obs_drop': args.obs_drop,
        'switch_min': args.switch_min, 'random_start': args.random_start, 'p_off': args.p_off,
        'setting_input': args.setting_input, 'out_threshold': args.out_threshold, 'reward_scale': args.reward_scale,
        'q_scale': policy_net[1].q_scale,
        'tau_reward': env.tau_reward, 'TMAX': env.TMAX, 'num_steps': env.num_steps,
        'BATCH_SIZE': base.BATCH_SIZE, 'GAMMA': base.GAMMA, 'EPS_START': base.EPS_START,
        'EPS_END': base.EPS_END, 'EPS_DECAY': base.EPS_DECAY, 'TAU': base.TAU, 'LR': base.LR,
        'BETA': base.BETA, 'device': str(base.device),
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    wandb.init(project='cl-dbs-rl-revision', name=args.run_name, config=config,
               dir=os.path.join(ROOT, 'wandb', '..'))

    # train() saves checkpoints/intermediate results to relative paths; run it from the run folder.
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        results = base.train(policy_net, target_net, memory, optimizer, env, args.num_episodes, 25,
                             base.device, curriculum=not args.no_curriculum,
                             pd_duration=args.pd_duration, healthy_duration=args.healthy_duration)
    finally:
        os.chdir(cwd)

    save_results(os.path.join(run_dir, 'training_results.pth'), results)
    torch.save(policy_net.state_dict(), os.path.join(run_dir, 'model.pth'))
    if getattr(env, 'step_log', None):
        np.savez_compressed(os.path.join(run_dir, 'step_log.npz'), rows=np.asarray(env.step_log, dtype=np.float64),
                            columns=np.array(['episode', 'step', 'PD', 'freq', 'pw', 'amp', 'beta', 'beta_smooth',
                                              'E', 'reward']))
    config.update({'finished': datetime.now().isoformat(timespec='seconds'),
                   'n_windows': env.n_windows, 'n_missing_beta': env.n_missing_beta})
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    wandb.finish()
    print(f'saved {run_dir}/model.pth (missing-beta windows: {env.n_missing_beta}/{env.n_windows})')


if __name__ == '__main__':
    main()
