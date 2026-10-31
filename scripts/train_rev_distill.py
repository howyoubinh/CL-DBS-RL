"""
Distillation: compress a teacher with a non-spiking Q readout (e.g. teachers_round4/RT4_p202) into
a sparse student with the revised readout and decoder (src/utils/rev_rl_utils.py).

The original script (train_distill_rockpool_16ch.py) distilled on one window per "epoch", taken
from env.reset() (unstimulated priming window) with a batch of one and the summed-over-batch Q,
so 100 epochs saw 100 unstimulated windows and no stimulated window. Here:

  collect   runs the teacher's own controller in the gym_pd_round3 environment (transitions, random
            starts, epsilon-greedy with --epsilon for coverage) on the training patient and saves
            every observation window with the teacher's Q (per-sample, q_scale applied):
            <out>/windows.npz  (X: (N, 100, 80) uint8, Q: (N, 9), PD, setting)
  train     fits a student of the teacher's architecture (non-spiking readout, same q_scale) on
            those windows: per-head KL(softmax(Q_t / T) || softmax(Q_s / T)) + lambda * KL-rate
            sparsity penalty on the hidden layers (target rho), minibatches, several epochs; reports
            per-head and joint action agreement with the teacher on a held-out 20% of windows and
            the hidden firing rates. Saves model.pth + config.json readable by run_rev_eval.py
            (controller model:<run_dir>).

  python scripts/train_rev_distill.py collect --teacher models/revision/teachers_round4/RT4_p202 \
      --episodes 100 --out data/revision/distill/RT4_p202
  python scripts/train_rev_distill.py train --teacher models/revision/teachers_round4/RT4_p202 \
      --windows data/revision/distill/RT4_p202/windows.npz --rho 0.015 --lam 1500 --temperature 4 \
      --run-name s0.015_w1500 --out-root models/revision/students_round4
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from scripts.train_rev_teacher import make_net  # noqa: E402
from src.environment.gym_pd_round3 import MousePDEnvRound3  # noqa: E402
from src.utils import rev_rl_utils as R  # noqa: E402

ENV_KW = dict(leap={'pw': 0.1, 'amp': 5.0, 'freq': 5.0}, num_steps=100, tau_reward=3000., delta=1.0,
              thresh_time_req=999, binning='hash')


def load_teacher(run_dir, device):
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    net = make_net(cfg.get('num_steps', 100), cfg.get('setting_input', False), cfg.get('out_threshold', 1.0), device,
                   cfg.get('linear_bias', True))
    net[1].q_scale = cfg.get('q_scale', 1.0)
    net.load_state_dict(torch.load(os.path.join(run_dir, 'model.pth'), map_location=device))
    net.eval()
    return net, cfg


def collect(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher, cfg = load_teacher(args.teacher, device)
    assert not cfg.get('setting_input'), 'collect assumes an 80-channel observation'
    rng = random.Random(args.seed)
    env = MousePDEnvRound3(tau_beta_max=cfg['tau_beta_max'], max_steps=args.max_steps, TMAX=cfg['window_ms'],
                           seed=cfg['patient_seed'], beta_smooth_windows=cfg.get('beta_smooth_windows', 10),
                           energy_norm=cfg.get('energy_norm', 'max'), transitions=True, random_start=True,
                           rng_seed=args.seed, reward_scale=cfg.get('reward_scale', 1.0),
                           obs_keep=cfg.get('obs_keep'), obs_noise_hz=cfg.get('obs_noise', 0.0), **ENV_KW)
    X, Q, PD, S = [], [], [], []
    for ep in range(args.episodes):
        env.set_pd_state(ep % 2)
        _, info = env.reset()
        obs = info['raw_spike_data']
        for t in range(args.max_steps):
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = R.q_values(teacher, x)[0][0].cpu().numpy()
            X.append(obs.astype(np.uint8)); Q.append(q); PD.append(env.PD); S.append((env.freq, env.pw, env.amp))
            if rng.random() < args.epsilon:
                act = env.action_space.sample()
            else:
                act = R.decode(torch.as_tensor(q))
            _, _, _, _, info = env.step(act)
            obs = info['raw_spike_data']
        print(f'episode {ep + 1}/{args.episodes}: {len(X)} windows', flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'windows.npz'), X=np.stack(X), Q=np.stack(Q).astype(np.float32),
                        PD=np.array(PD), setting=np.array(S), meta=json.dumps({'teacher': args.teacher, 'episodes': args.episodes,
                                                                                'max_steps': args.max_steps, 'epsilon': args.epsilon,
                                                                                'seed': args.seed, 'patient_seed': cfg['patient_seed']}))
    print(f'saved {args.out}/windows.npz: {len(X)} windows, PD share {np.mean(PD):.2f}, stimulated share '
          f'{np.mean([s[0] > 0 and s[2] > 0 for s in S]):.2f}')


def head_kl(q_t, q_s, T):
    """Sum over the three heads of KL(softmax(q_t/T) || softmax(q_s/T)) * T^2, batch mean."""
    loss = 0.0
    for j in range(3):
        pt = F.softmax(q_t[:, 3 * j:3 * j + 3] / T, dim=1)
        ls = F.log_softmax(q_s[:, 3 * j:3 * j + 3] / T, dim=1)
        loss = loss + F.kl_div(ls, pt, reduction='batchmean') * T * T
    return loss


def centred_mse(q_t, q_s):
    """MSE between head-centred Q (Q minus its mean within each 3-action head): regresses the
    action gaps directly, invariant to the large common offset of the teacher's Q."""
    loss = 0.0
    for j in range(3):
        a, b = q_t[:, 3 * j:3 * j + 3], q_s[:, 3 * j:3 * j + 3]
        loss = loss + F.mse_loss(b - b.mean(1, keepdim=True), a - a.mean(1, keepdim=True))
    return loss


def agreement(q_t, q_s):
    a_t = torch.stack([q_t[:, 3 * j:3 * j + 3].argmax(1) for j in range(3)], 1)
    a_s = torch.stack([q_s[:, 3 * j:3 * j + 3].argmax(1) for j in range(3)], 1)
    per_head = (a_t == a_s).float().mean(0)
    return per_head.tolist(), float((a_t == a_s).all(1).float().mean())


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    teacher, cfg = load_teacher(args.teacher, device)
    d = np.load(args.windows)
    X = torch.as_tensor(d['X'], dtype=torch.float32)
    Q = torch.as_tensor(d['Q'])
    n = len(X); idx = np.random.permutation(n); n_val = n // 5
    va, tr = idx[:n_val], idx[n_val:]

    run_dir = os.path.join(args.out_root, args.run_name)
    if os.path.exists(os.path.join(run_dir, 'model.pth')):
        sys.exit(f'{run_dir}/model.pth exists; refusing to overwrite')
    os.makedirs(run_dir, exist_ok=True)

    student = make_net(cfg.get('num_steps', 100), False, cfg.get('out_threshold', 1.0), device, not args.no_linear_bias,
                       'xylo' if args.xylo_taus else None)
    assert not (args.init_from_teacher and args.no_linear_bias), '--init-from-teacher needs the teacher architecture (biases)'
    student[1].q_scale = cfg.get('q_scale', 1.0)
    student[1].target_sparsity, student[1].sparsity_weight = args.rho, args.lam
    if args.init_from_teacher:
        student.load_state_dict(teacher.state_dict())
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    def evaluate(ids):
        student.eval()
        qs, rates = [], []
        with torch.no_grad():
            for b in range(0, len(ids), 512):
                j = ids[b:b + 512]
                R.reset_state(student)
                _, mem, _, rate, rec = student(X[j].to(device))
                qs.append(mem.sum(1).cpu() / student[1].q_scale)
                rates.append(float(rec['1_LIFTorch']['spikes'].float().mean() + rec['3_LIFTorch']['spikes'].float().mean()) / 2)
        q_s = torch.cat(qs)
        per_head, joint = agreement(Q[ids], q_s)
        student.train()
        return per_head, joint, float(np.mean(rates)), float(head_kl(Q[ids], q_s, args.temperature))

    history = []
    for epoch in range(args.epochs):
        perm = np.random.permutation(tr)
        sched = min(1.0, 2.0 * epoch / max(1, args.epochs))        # sparsity ramps in over the first half, as published
        tot = 0.0
        for b in range(0, len(perm), args.batch):
            j = perm[b:b + args.batch]
            q_s, pen, _ = R.q_values(student, X[j].to(device))
            q_t = Q[j].to(device)
            loss = head_kl(q_t, q_s, args.temperature) + args.mse_weight * centred_mse(q_t, q_s) + sched * pen
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); tot += float(loss) * len(j)
        per_head, joint, rate, kl = evaluate(va)
        history.append({'epoch': epoch, 'train_loss': tot / len(tr), 'val_kl': kl, 'val_agree_heads': per_head,
                        'val_agree_joint': joint, 'hidden_rate': rate})
        print(f'epoch {epoch + 1}/{args.epochs}: loss {tot / len(tr):.4f}  val KL {kl:.4f}  agreement freq/pw/amp '
              f'{per_head[0]:.2f}/{per_head[1]:.2f}/{per_head[2]:.2f} joint {joint:.2f}  hidden rate {rate:.4f}', flush=True)

    torch.save(student.state_dict(), os.path.join(run_dir, 'model.pth'))
    t_rate = None
    with torch.no_grad():
        R.reset_state(teacher)
        _, _, _, _, rec = teacher(X[va[:512]].to(device))
        t_rate = float(rec['1_LIFTorch']['spikes'].float().mean() + rec['3_LIFTorch']['spikes'].float().mean()) / 2
    out_cfg = {'run_name': args.run_name, 'argv': sys.argv, 'finished': datetime.now().isoformat(timespec='seconds'),
               'teacher': args.teacher, 'windows': args.windows, 'n_windows': int(n), 'rho': args.rho, 'lam': args.lam,
               'temperature': args.temperature, 'mse_weight': args.mse_weight, 'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
               'init_from_teacher': args.init_from_teacher, 'seed': args.seed,
               # fields run_rev_eval.py reads
               'window_ms': cfg['window_ms'], 'binning': cfg.get('binning', 'hash'), 'setting_input': False,
               'out_threshold': cfg.get('out_threshold', 1.0), 'q_scale': cfg.get('q_scale', 1.0), 'linear_bias': not args.no_linear_bias, 'taus': 'xylo' if args.xylo_taus else None, 'obs_keep': cfg.get('obs_keep'),
               'tau_beta_max': cfg['tau_beta_max'], 'patient_seed': cfg['patient_seed'],
               'final': history[-1], 'teacher_hidden_rate': t_rate, 'history': history}
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(out_cfg, f, indent=2)
    print(f'saved {run_dir}/model.pth  (teacher hidden rate {t_rate:.4f}, student {history[-1]["hidden_rate"]:.4f}, '
          f'joint agreement {history[-1]["val_agree_joint"]:.2f})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--teacher', required=True)
    c.add_argument('--episodes', type=int, default=100)
    c.add_argument('--max-steps', type=int, default=100)
    c.add_argument('--epsilon', type=float, default=0.1)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--out', required=True)
    t = sub.add_parser('train')
    t.add_argument('--teacher', required=True)
    t.add_argument('--windows', required=True)
    t.add_argument('--rho', type=float, required=True)
    t.add_argument('--lam', type=float, required=True)
    t.add_argument('--temperature', type=float, default=4.0)
    t.add_argument('--mse-weight', type=float, default=0.0, help='weight of the head-centred Q MSE term')
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch', type=int, default=128)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--init-from-teacher', action='store_true')
    t.add_argument('--no-linear-bias', action='store_true', help='Xylo-native student: no biases on the linear layers')
    t.add_argument('--xylo-taus', action='store_true', help='Xylo-native student: bit-shift-exact time constants')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--run-name', required=True)
    t.add_argument('--out-root', default=os.path.join(ROOT, 'models/revision/students_round4'))
    args = ap.parse_args()
    (collect if args.cmd == 'collect' else train)(args)


if __name__ == '__main__':
    main()
