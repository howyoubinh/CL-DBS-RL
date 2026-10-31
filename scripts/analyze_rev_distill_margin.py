"""Student-teacher action agreement on held-out distillation windows, overall and where the teacher's
choice is clear (gap of at least 0.15 between its two best actions, for every parameter).

Reproduces the joint agreement stored in each student's config.json (same validation split), then adds
the clear-margin agreement and the teacher's action gaps. Writes data/revision/distill/margin_agreement.json.
"""
import glob, json, os, sys
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from scripts.train_rev_teacher import make_net
from src.utils import rev_rl_utils as R

MARGIN = 0.15
d = np.load('data/revision/distill/RT4_p202/windows.npz')
X = torch.as_tensor(d['X'], dtype=torch.float32); Q = torch.as_tensor(d['Q']).float()


def heads(q):
    return torch.stack([q[:, 3 * j:3 * j + 3].argmax(1) for j in range(3)], 1)


def gaps(q):
    s = [q[:, 3 * j:3 * j + 3].sort(1, descending=True).values for j in range(3)]
    return torch.stack([h[:, 0] - h[:, 1] for h in s], 1)


out = {'margin': MARGIN, 'students': {}}
g_all = gaps(Q)
out['teacher_gap_median_per_head'] = [float(g_all[:, j].median()) for j in range(3)]
out['teacher_gap_median_pooled'] = float(g_all.median())
for cfg in sorted(glob.glob('models/revision/students_round4/*/config.json')):
    c = json.load(open(cfg)); name = os.path.basename(os.path.dirname(cfg))
    np.random.seed(c.get('seed', 0)); torch.manual_seed(c.get('seed', 0))
    va = np.random.permutation(len(X))[:len(X) // 5]
    net = make_net(c.get('num_steps', 100), False, c.get('out_threshold', 1.0), 'cpu', c.get('linear_bias', True),
                   'xylo' if c.get('taus') == 'xylo' else None)
    net[1].q_scale = c.get('q_scale', 1.0)
    net.load_state_dict(torch.load(os.path.dirname(cfg) + '/model.pth', map_location='cpu')); net.eval()
    qs = []
    with torch.no_grad():
        for b in range(0, len(va), 512):
            j = va[b:b + 512]; R.reset_state(net); _, mem, _, _, _ = net(X[j]); qs.append(mem.sum(1) / net[1].q_scale)
    qs = torch.cat(qs); qt = Q[va]
    hit = heads(qt) == heads(qs); g = gaps(qt); clear = (g >= MARGIN).all(1)
    out['students'][name] = {
        'joint': float(hit.all(1).float().mean()),
        'joint_stored': c.get('final', {}).get('val_agree_joint'),
        'clear_fraction': float(clear.float().mean()),
        'joint_clear': float(hit[clear].all(1).float().mean()),
        'per_head_clear': [float(hit[g[:, j] >= MARGIN, j].float().mean()) for j in range(3)],
    }
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out['students'][name].items()})
json.dump(out, open('data/revision/distill/margin_agreement.json', 'w'), indent=1)
print('wrote data/revision/distill/margin_agreement.json')
