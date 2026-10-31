"""
Matched non-spiking baseline training (MLP, GRU, LSTM).

Trains a non-spiking DQN baseline with the same training loop, hyperparameters, reward,
environment and curriculum as the SNN teacher. It reuses train() from
scripts/train_rnn_baseline.py unchanged (model-agnostic: (batch, time, 80) -> 9 Q-values), with the
environments of scripts/train_rev_teacher.py: reward threshold, window/binning and the simulated
patient are flags, and episode settings default to the teacher's (50 steps per episode, 100/100
curriculum; the original RNN/DRQN baselines were trained with 100-step episodes).

  --arch        mlp | gru | lstm
  --input       mean80 (published ANN) | raw80 (published GRU) | pool16 (the SNN's exact input)
                | mean16 (MLP only: the SNN's layer sizes, 16 -> 128 -> 128 -> 9; hardware comparison)
  --lr, --hidden  exposed for the hyperparameter sweep (published: 1e-3, 128)
  --env round3  the gym_pd_round3 environment and curriculum: reward on the 1 s mean of beta,
                energy / max RMS, --transitions, --random-start, --reward-scale, exactly as the RT4
                teachers (scripts/train_rev_teacher.py); the SNN readout options (--out-threshold,
                --q-scale) do not apply (these models have linear outputs). Saves step_log.npz.

  python scripts/train_rev_baseline.py --arch gru --input raw80 --patient-seed 101 --run-name gru_raw80_p101
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import torch
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import wandb  # noqa: E402

import scripts.train_rnn_baseline as base  # noqa: E402  (train() + hyperparameters)
from src.environment.gym_pd_rev import MousePDEnvAdaptiveRev  # noqa: E402
from src.environment.gym_pd_round3 import MousePDEnvRound3  # noqa: E402
from src.environment.replay_memory import ReplayMemory  # noqa: E402
from src.models.rev_baselines import build  # noqa: E402
from src.utils.io_utils import save_results  # noqa: E402

SNAPSHOT = ['scripts/train_rev_baseline.py', 'scripts/train_rnn_baseline.py', 'src/models/rev_baselines.py',
            'src/utils/rnn_utils.py', 'src/environment/gym_pd.py', 'src/environment/gym_pd_rev.py',
            'src/environment/gym_pd_round3.py',
            'src/simulation/simulate_network_optimized.py', 'src/utils/data_processing.py']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arch', choices=['mlp', 'gru', 'lstm'], required=True)
    ap.add_argument('--input', choices=['mean80', 'mean16', 'raw80', 'pool16'], required=True)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--tau-beta', type=float, default=150.0)
    ap.add_argument('--window-ms', type=int, default=100)
    ap.add_argument('--binning', choices=['hash', 'count'], default='hash')
    ap.add_argument('--patient-seed', type=int, required=True)
    ap.add_argument('--agent-seed', type=int, default=None)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--out-root', default=os.path.join(ROOT, 'models/revision/baselines'))
    ap.add_argument('--delta', type=float, default=1.0)
    ap.add_argument('--num-episodes', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=50)
    ap.add_argument('--pd-duration', type=int, default=100)
    ap.add_argument('--healthy-duration', type=int, default=100)
    ap.add_argument('--env', choices=['rev', 'round3'], default='rev')
    ap.add_argument('--beta-smooth', type=int, default=10)
    ap.add_argument('--energy-norm', default='max')
    ap.add_argument('--transitions', action='store_true')
    ap.add_argument('--switch-min', type=int, default=10)
    ap.add_argument('--random-start', action='store_true')
    ap.add_argument('--p-off', type=float, default=0.25)
    ap.add_argument('--reward-scale', type=float, default=1.0)
    args = ap.parse_args()
    if args.arch == 'mlp' and args.input == 'raw80':
        sys.exit('mlp takes mean80, mean16 or pool16')
    if args.arch != 'mlp' and args.input in ('mean80', 'mean16'):
        sys.exit('recurrent models take raw80 or pool16')
    if args.window_ms != 100 and args.binning != 'count':
        sys.exit('windows other than 100 ms need --binning count')
    agent_seed = args.patient_seed if args.agent_seed is None else args.agent_seed

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
                               reward_scale=args.reward_scale, **env_kw)
    else:
        env = MousePDEnvAdaptiveRev(**env_kw)
    policy_net = build(args.arch, args.input, args.hidden).to(base.device)
    target_net = build(args.arch, args.input, args.hidden).to(base.device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
    memory = ReplayMemory(100000)

    try:
        commit = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = None
    config = {
        'run_name': args.run_name, 'argv': sys.argv, 'started': datetime.now().isoformat(timespec='seconds'),
        'git_commit_public_repo': commit, 'arch': args.arch, 'input': args.input, 'lr': args.lr,
        'hidden': args.hidden, 'window_ms': args.window_ms, 'binning': args.binning,
        'tau_beta_max': args.tau_beta, 'patient_seed': args.patient_seed, 'agent_seed': agent_seed,
        'delta': args.delta, 'num_episodes': args.num_episodes, 'max_steps': args.max_steps,
        'pd_duration': args.pd_duration, 'healthy_duration': args.healthy_duration,
        'env_class': env.__class__.__name__, 'env': args.env,
        'beta_smooth_windows': getattr(env, 'beta_smooth_windows', 1),
        'energy_norm': args.energy_norm if args.env == 'round3' else '250', 'E_max': getattr(env, 'E_max', 250.0),
        'transitions': args.transitions, 'switch_min': args.switch_min, 'random_start': args.random_start,
        'p_off': args.p_off, 'reward_scale': args.reward_scale, 'BATCH_SIZE': base.BATCH_SIZE, 'GAMMA': base.GAMMA,
        'EPS_START': base.EPS_START, 'EPS_END': base.EPS_END, 'EPS_DECAY': base.EPS_DECAY, 'TAU': base.TAU,
        'n_params': sum(p.numel() for p in policy_net.parameters()), 'device': str(base.device),
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    wandb.init(project='cl-dbs-rl-revision', name=args.run_name, config=config, dir=ROOT)

    cwd = os.getcwd()
    os.chdir(run_dir)                       # train() writes checkpoints to relative paths
    try:
        results = base.train(policy_net, target_net, memory, optimizer, env, args.num_episodes, 25, base.device,
                             curriculum=True, pd_duration=args.pd_duration, healthy_duration=args.healthy_duration)
    finally:
        os.chdir(cwd)
    save_results(os.path.join(run_dir, 'training_results.pth'), results)
    torch.save(policy_net.state_dict(), os.path.join(run_dir, 'model.pth'))
    if getattr(env, 'step_log', None):
        import numpy as np
        np.savez_compressed(os.path.join(run_dir, 'step_log.npz'), rows=np.asarray(env.step_log, dtype=np.float64),
                            columns=np.array(['episode', 'step', 'PD', 'freq', 'pw', 'amp', 'beta', 'beta_smooth',
                                              'E', 'reward']))
    config.update({'finished': datetime.now().isoformat(timespec='seconds'),
                   'n_windows': env.n_windows, 'n_missing_beta': env.n_missing_beta})
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    wandb.finish()
    print(f'saved {run_dir}/model.pth')


if __name__ == '__main__':
    main()
