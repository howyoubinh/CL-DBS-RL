"""
Runner for the experiment manifest (scripts/rev_manifest.yaml).

Each manifest entry describes one experiment: a command template, a parameter grid that
expands into jobs, an output folder, and optional analysis commands to run after all jobs
finish. Every job writes its own log; every invocation appends a record (time, public-repo
commit, jobs, exit codes) to <outputs>/RUNS.jsonl, so any result can be traced to the exact
commands that made it.

  python scripts/run_rev.py list                 # entries, status, description
  python scripts/run_rev.py show final_cycling     # expanded job commands
  python scripts/run_rev.py run final_cycling [--dry-run] [--parallel 22] [--only-missing] [--match RE] [--exclude RE]
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MANIFEST = os.path.join(ROOT, 'scripts', 'rev_manifest.yaml')
ENV = {'OMP_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'WANDB_MODE': 'offline'}
# Shell prefix for every job. Set CLDBS_CONDA_ENV to the conda environment that holds the
# dependencies (default: the one from environment.yml), or CLDBS_PRELUDE to replace the whole prefix.
CONDA_ENV = os.environ.get('CLDBS_CONDA_ENV', 'cl-dbs-rl')
PRELUDE = os.environ.get('CLDBS_PRELUDE', f'eval "$(conda shell.bash hook)" && conda activate {CONDA_ENV} && ')


def load():
    with open(MANIFEST) as f:
        return yaml.safe_load(f)


def expand(entry):
    """Expand the grid into (job_name, command, expected_output or None)."""
    jobs = []
    for block in entry.get('grid', [{}]):
        block = dict(block)
        block_expect = block.pop('_expect', None)  # per-block override of entry['expect']
        keys = list(block)
        for values in itertools.product(*(block[k] if isinstance(block[k], list) else [block[k]] for k in keys)):
            params = dict(zip(keys, values))
            params['out'] = entry['outputs']
            cmd = entry['template'].format(**params)
            name = '_'.join(str(params[k]).strip('-').replace(' ', '') for k in keys if str(params[k])) or 'job'
            name = name.replace('models/revision/', '').replace('/', '_')   # model paths: flat log names
            pattern = block_expect or entry.get('expect')
            expect = pattern.format(**params) if pattern else None
            jobs.append((name, cmd, expect))
    return jobs


def run_job(name, cmd, logdir, gpu=None):
    env = dict(os.environ, **ENV)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with open(os.path.join(logdir, f'{name}.log'), 'w') as log:
        rc = subprocess.call(['bash', '-c', PRELUDE + cmd], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f'{"done  " if rc == 0 else "FAILED"} {name}', flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['list', 'show', 'run'])
    ap.add_argument('entry', nargs='?')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--parallel', type=int, default=None)
    ap.add_argument('--only-missing', action='store_true', help='skip jobs whose expected output exists')
    ap.add_argument('--match', default=None, help='keep only jobs whose command contains this regex')
    ap.add_argument('--exclude', default=None, help='drop jobs whose command contains this regex')
    args = ap.parse_args()
    m = load()

    if args.action == 'list':
        for k, e in m.items():
            print(f'{k:28s} {e.get("status", "planned"):8s} {e.get("description", "")[:70]}')
        return
    e = m[args.entry]
    out = os.path.join(ROOT, e['outputs'])
    jobs = expand(e) if 'template' in e else []
    if args.only_missing:
        jobs = [j for j in jobs if not (j[2] and os.path.exists(os.path.join(ROOT, j[2])))]
    if args.match:
        jobs = [j for j in jobs if re.search(args.match, j[1])]
    if args.exclude:
        jobs = [j for j in jobs if not re.search(args.exclude, j[1])]
    if args.action == 'show' or args.dry_run:
        for name, cmd, _ in jobs:
            print(f'[{name}] {cmd}')
        for cmd in e.get('after', []):
            print(f'[after] {cmd.format(out=e["outputs"])}')
        return

    logdir = os.path.join(out, 'logs')
    os.makedirs(logdir, exist_ok=True)
    par = args.parallel or e.get('parallel', 8)
    gpus = e.get('gpus')  # e.g. [0, 1]: jobs alternate between them
    with ThreadPoolExecutor(par) as pool:
        futs = [pool.submit(run_job, n, c, logdir, gpus[i % len(gpus)] if gpus else None)
                for i, (n, c, _) in enumerate(jobs)]
        rcs = [f.result() for f in futs]
    after_rc = []
    if all(rc == 0 for rc in rcs):
        for i, cmd in enumerate(e.get('after', [])):
            after_rc.append(run_job(f'after_{i}', cmd.format(out=e['outputs']), logdir))
    try:
        commit = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = None
    with open(os.path.join(out, 'RUNS.jsonl'), 'a') as f:
        f.write(json.dumps({'time': datetime.now().isoformat(timespec='seconds'), 'entry': args.entry,
                            'public_repo_commit': commit, 'jobs': [{'name': n, 'cmd': c, 'rc': rc}
                            for (n, c, _), rc in zip(jobs, rcs)], 'after_rc': after_rc}) + '\n')
    print(f'{sum(rc == 0 for rc in rcs)}/{len(rcs)} jobs succeeded; record appended to {e["outputs"]}/RUNS.jsonl')
    sys.exit(0 if all(rc == 0 for rc in rcs + after_rc) else 1)


if __name__ == '__main__':
    main()
