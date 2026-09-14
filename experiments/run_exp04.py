"""Reproduce EXP04 training; nonempty run directories are never overwritten."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--confirm', action='store_true', help='Also reproduce the frozen event c=2 seed43/44 comparisons')
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from baseline import sha256, write_json

protocol = root/'runs/exp04_protocol'
if protocol.exists():
    raise ValueError('Protocol already exists; reproduce in a fresh project directory')
baseline = root/'runs/baseline_v1'
if not (baseline/'manifest.json').exists():
    raise ValueError('The fixed runs/baseline_v1 reference must be present')
protocol.mkdir(parents=True)
write_json(protocol/'baseline_hashes.json', {str(f.relative_to(root)):sha256(f) for f in baseline.rglob('*') if f.is_file()})
write_json(protocol/'protocol.json', dict(candidates=['class_c2','class_c4','event_c2','event_c4'],
           selection='fold0 thresholds -> fold1 score, ties c2; freeze before fold2 analysis',
           singleton='c', seed=42, rounds=300, threads=4, features='raw', parent='baseline_v1'))
source = protocol/'source'
source.mkdir()
for name in ['baseline.py','train.py','predict.py','metrics.py','requirements.txt']:
    shutil.copy2(root/name, source/name)
shutil.copytree(root/'experiments', source/'experiments', ignore=shutil.ignore_patterns('__pycache__'))
python = str(root/'.venv/bin/python')
jobs = [('none', 2, 'runs/exp04_baseline_control')] + [
    (mode,c,f'runs/exp04{letter}_{name}/c{c}')
    for mode,letter,name in [('class','a','class_weight'),('event','b','event_weight')]
    for c in (2,4)]
for mode,c,out in jobs:
    cmd=[python,'train.py','--output',out,'--training-weight',mode,'--positive-weight',str(c),'--threads','4','--rounds','300']
    print('RUN', ' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=root,check=True)
subprocess.run([python,'experiments/exp04.py','select'],cwd=root,check=True)
subprocess.run([python,'experiments/exp04.py','diagnose'],cwd=root,check=True)
if args.confirm:
    for seed in (43,44):
        for mode in ('none','event'):
            subprocess.run([python,'train.py','--output',f'runs/exp04_confirmation/{mode}_seed{seed}',
                            '--training-weight',mode,'--positive-weight','2','--threads','4',
                            '--rounds','300','--seed',str(seed)],cwd=root,check=True)
    subprocess.run([python,'experiments/confirm_exp04.py'],cwd=root,check=True)
