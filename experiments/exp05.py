"""Independent single-factor EXP05, chronological selection before fold2 analysis."""
import json
import subprocess
import sys
from pathlib import Path
import shutil
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline import sha256, write_json, event_safe_boundary
from metrics import aggregate, choose_threshold, score, event_weights

OUT = ROOT / 'runs/exp05_analysis'
PROTOCOL = ROOT / 'runs/exp05_protocol'
CONFIGS = {
    'baseline': {},
    'leaf20': {'min-data-in-leaf': 20}, 'leaf50': {'min-data-in-leaf': 50},
    'leaves7': {'num-leaves': 7}, 'leaves31': {'num-leaves': 31},
    'rounds100': {'rounds': 100}, 'rounds600': {'rounds': 600},
}
RUNS = {
    'baseline': 'exp05_baseline_control',
    'leaf20': 'exp05a_leaf_samples/leaf20', 'leaf50': 'exp05a_leaf_samples/leaf50',
    'leaves7': 'exp05b_num_leaves/leaves7', 'leaves31': 'exp05b_num_leaves/leaves31',
    'rounds100': 'exp05c_rounds/rounds100', 'rounds600': 'exp05c_rounds/rounds600',
}

def train(name, config, seed=42, resume=False):
    folder = ROOT / 'runs' / (RUNS[name] if seed == 42 else f'exp05_confirmation/{name}_seed{seed}')
    if resume and (folder/'manifest.json').exists():
        meta = json.loads((folder/'manifest.json').read_text())
        expected = dict(rounds=300,min_data_in_leaf=100,num_leaves=15,seed=seed)
        expected.update({k.replace('-','_'):v for k,v in config.items()})
        assert all(meta['config'][k] == v for k,v in expected.items())
        assert (folder/'metrics.json').exists()
        assert len(list((folder/'validation').glob('*.csv'))) == 3*len(meta['devices'])
        print(f'Reused completed {name} seed={seed}',flush=True)
        return folder
    cmd = [sys.executable, 'train.py', '--output', str(folder), '--seed', str(seed)]
    for k, v in config.items():
        cmd += ['--' + k, str(v)]
    with (PROTOCOL / f'{name}_seed{seed}.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Trained {name} seed={seed}', flush=True)
    return folder

def load(folder, fold):
    # Do not read metrics.json here: it includes the later diagnostic scores.
    return {p.name.split('_fold')[0]: pd.read_csv(p) for p in sorted((folder/'validation').glob(f'*_fold{fold}.csv'))}

def thresholds(folder, folds):
    blocks = {d: [] for d in load(folder, folds[0])}
    for f in folds:
        for d, df in load(folder, f).items():
            if (folder/'models'/f'{d}_fold{f}.txt').exists():
                blocks[d].append(dict(y=df.label.to_numpy(), p=df.probability.to_numpy()))
    pooled = [b for bs in blocks.values() for b in bs]
    fallback = choose_threshold(pooled)
    result = {}
    for d, bs in blocks.items():
        events = sum(int((np.diff(np.r_[0,b['y'],0]) == 1).sum()) for b in bs)
        enough = events >= 2 and any((b['y'] == 0).any() for b in bs)
        result[d] = (choose_threshold(bs) if enough else fallback,
                     'device_calibration' if enough else 'pooled_calibration', events)
    return result

def segments(y):
    edges = np.diff(np.r_[0,y,0])
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def evaluate(folder, name, fold, calibration, section=None):
    ts = thresholds(folder, calibration)
    devices, events, alarms, parts = [], [], [], []
    for d, df in load(folder, fold).items():
        if section:
            split = event_safe_boundary(df.label.to_numpy(), int(pd.DatetimeIndex(pd.to_datetime(df.timestamp)).searchsorted('2024-08-01')))
            df = df.iloc[:split] if section == 'july' else df.iloc[split:]
        y = df.label.to_numpy(); t, source, count = ts[d]
        p = (df.probability.to_numpy() >= t).astype(np.int8)
        s = score(y,p); parts.append(s)
        devices.append(dict(device=d, **s, fp=int(((y==0)&(p==1)).sum()), fn=int(((y==1)&(p==0)).sum()),
                            threshold=t, threshold_source=source, calibration_events=count))
        w = event_weights(y)
        for a,b in segments(y):
            hit = np.flatnonzero(p[a:b]); length = b-a
            events.append(dict(device=d,start=df.timestamp.iloc[a],end=df.timestamp.iloc[b-1],points=length,
                missed_event=not len(hit),delay_minutes=int(hit[0])*20 if len(hit) else None,
                recall_3h=float(p[a:min(b,a+9)].mean()), recall_6h=float(p[a:min(b,a+18)].mean()),
                weighted_recall=float((w[a:b]*p[a:b]).sum()/w[a:b].sum())))
        for a,b in segments(((y==0)&(p==1)).astype(np.int8)):
            alarms.append(dict(device=d,start=df.timestamp.iloc[a],end=df.timestamp.iloc[b-1],points=b-a,duration_minutes=(b-a)*20))
    tag = section or ('forward_fold1' if fold == 1 else 'diagnostic_fold2')
    dest = OUT/name; dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(devices).to_csv(dest/f'{tag}_devices.csv',index=False)
    pd.DataFrame(events).to_csv(dest/f'{tag}_events.csv',index=False)
    pd.DataFrame(alarms,columns=['device','start','end','points','duration_minutes']).to_csv(dest/f'{tag}_false_alarm_segments.csv',index=False)
    result = dict(aggregate=aggregate(parts), false_alarm_segments=len(alarms),
        long_false_alarm_points=int(sum(x['points'] for x in alarms if x['points']>=18)),
        alarm_duration_minutes_quantiles=np.quantile([x['duration_minutes'] for x in alarms],[0,.5,.9,1]).tolist() if alarms else [],
        missed_events=sum(x['missed_event'] for x in events), events=len(events))
    write_json(dest/f'{tag}.json',result)
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resume',action='store_true',help='Reuse completed matching runs after an interrupted analysis')
    args=ap.parse_args()
    if not args.resume and (OUT.exists() or PROTOCOL.exists()):
        raise SystemExit('Output already exists; refusing to overwrite.')
    if args.resume:
        assert (PROTOCOL/'protocol.json').exists() and not (OUT/'selection.json').exists(), 'Only resume before selection has been locked.'
        for filename in ['baseline.py','train.py','metrics.py']:
            assert sha256(ROOT/filename)==sha256(PROTOCOL/'source'/filename), 'Training source changed.'
    OUT.mkdir(exist_ok=True); PROTOCOL.mkdir(exist_ok=True)
    if not args.resume:
        write_json(PROTOCOL/'protocol.json', dict(candidates=CONFIGS,parent='baseline_v1',seed=42,
        selection='fold0 eligible supervised probabilities calibrate; fold1 selects best including baseline per family; ties prefer baseline; freeze before any fold2 analysis',
        confirmation='Each nonbaseline family winner with positive fold1 gain gets paired seeds 43 and 44; adoption requires positive fold1 and fold2 gains across all seeds and no negative July/August aggregate gains at seed42.',
        evaluation='fold2 is previously inspected development diagnosis, not an untouched test. July/August are event-safe subblocks of fold1 with same past-only trained model and fold0 thresholds.',
        invariants='raw features, unit training weights, original threshold rule, fixed folds and gap, no early stopping; no other experiment stacked'))
    baseline_files = {str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'runs/baseline_v1').rglob('*') if p.is_file()}
    source = PROTOCOL/'source'
    if not args.resume:
        write_json(PROTOCOL/'baseline_hashes.json',baseline_files)
        (PROTOCOL/'git_head.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
        (PROTOCOL/'changes.patch').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
        source.mkdir()
        for filename in ['baseline.py','train.py','metrics.py','predict.py','experiments/exp05.py']:
            shutil.copy2(ROOT/filename,source/Path(filename).name)
    else:
        assert baseline_files == json.loads((PROTOCOL/'baseline_hashes.json').read_text())
    folders = {name:train(name,cfg,resume=args.resume) for name,cfg in CONFIGS.items()}
    forward = {n:evaluate(f,n,1,[0]) for n,f in folders.items()}
    winners = {}
    for family, names in {'A':['baseline','leaf20','leaf50'],'B':['baseline','leaves7','leaves31'],'C':['baseline','rounds100','rounds600']}.items():
        winners[family] = max(names,key=lambda n:forward[n]['aggregate']['score'])
    write_json(OUT/'selection.json',dict(winners=winners,forward=forward,rule='fold0 -> fold1 only; baseline participates'))
    print('Selection locked: '+json.dumps(winners),flush=True)
    selected = sorted(set(winners.values())-{'baseline'})
    confirmations = {}
    for seed in [43,44] if selected else []:
        for n in ['baseline']+selected:
            f = train(n,CONFIGS[n],seed)
            confirmations[f'{n}_seed{seed}'] = dict(folder=str(f),forward=evaluate(f,f'{n}_seed{seed}',1,[0]),
                diagnostic=evaluate(f,f'{n}_seed{seed}',2,[0,1]))
    write_json(OUT/'confirmation.json',confirmations)
    results = {}
    for n,f in folders.items():
        m = json.loads((f/'metrics.json').read_text())
        results[n] = dict(config=CONFIGS[n],folder=str(f),forward=forward[n],diagnostic=evaluate(f,n,2,[0,1]),
            july=evaluate(f,n,1,[0],'july'),august=evaluate(f,n,1,[0],'august'),
            resources=dict(elapsed_seconds=m['elapsed_seconds'],peak_rss_mib=m['peak_rss_mib'],
                fit_seconds=sum(r['fit_seconds'] for r in m['folds'])+sum(d['final_fit_seconds'] for d in m['devices'].values()),
                validation_predict_seconds=sum(r['predict_seconds'] for r in m['folds'])))
    write_json(OUT/'results.json',results)
    control = folders['baseline']
    diffs = []
    for old in (ROOT/'runs/baseline_v1/validation').glob('*.csv'):
        a,b = pd.read_csv(old),pd.read_csv(control/'validation'/old.name)
        np.testing.assert_array_equal(a[['timestamp','label','prediction']].to_numpy(), b[['timestamp','label','prediction']].to_numpy())
        diffs.append(float(np.max(np.abs(a.probability-b.probability))))
    assert all(sha256(ROOT/p)==h for p,h in baseline_files.items())
    write_json(OUT/'verification.json',dict(baseline_unchanged=True,baseline_max_probability_difference=max(diffs),
        baseline_predictions_match=True,source_hashes={p.name:sha256(p) for p in source.iterdir()}))
    print('Completed EXP05 training and analysis.',flush=True)

if __name__ == '__main__':
    main()
