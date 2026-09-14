"""EXP04 analysis: freeze c on forward fold1 before reporting diagnostic fold2."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import write_json, sha256
from metrics import aggregate, choose_threshold, score

ROOT = Path('runs/exp04_analysis')
RUNS = {'baseline': Path('runs/baseline_v1')}
RUNS.update({f'{mode}_c{c}': Path(f'runs/exp04{letter}_{name}/c{c}')
             for mode, letter, name in [('class', 'a', 'class_weight'), ('event', 'b', 'event_weight')]
             for c in (2, 4)})


def segments(y):
    edges = np.diff(np.r_[0, np.asarray(y, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def thresholds_from_fold0(run):
    report = json.loads((run/'metrics.json').read_text())
    blocks = {}
    for device in report['devices']:
        row = next(r for r in report['folds'] if r['device'] == device and r['fold'] == 0)
        df = pd.read_csv(run/'validation'/f'{device}_fold0.csv')
        blocks[device] = [dict(y=df.label.to_numpy(), p=df.probability.to_numpy())] if row['model'] == 'lightgbm' else []
    pooled = [b for bs in blocks.values() for b in bs]
    fallback = choose_threshold(pooled)
    result = {}
    for device, bs in blocks.items():
        events = sum(len(segments(b['y'])) for b in bs)
        enough = events >= 2 and any((b['y'] == 0).any() for b in bs)
        result[device] = dict(threshold=choose_threshold(bs) if enough else fallback,
                              source='device_fold0' if enough else 'pooled_fold0', eligible_events=events)
    return result


def evaluate(name, fold, thresholds):
    run = RUNS[name]
    report = json.loads((run/'metrics.json').read_text())
    devices, events, false_alarms, parts = [], [], [], []
    for device, meta in thresholds.items():
        df = pd.read_csv(run/'validation'/f'{device}_fold{fold}.csv')
        y = df.label.to_numpy()
        pred = (df.probability.to_numpy() >= meta['threshold']).astype(np.int8)
        s = score(y, pred)
        parts.append(s)
        fp = ((y == 0) & (pred == 1))
        lengths = []
        for start, end in segments(fp):
            lengths.append(end-start)
            false_alarms.append(dict(device=device, start=df.timestamp.iloc[start], end=df.timestamp.iloc[end-1],
                                     points=end-start, hours=(end-start)/3))
        row = next(r for r in report['folds'] if r['device'] == device and r['fold'] == fold)
        devices.append(dict(device=device, **s, fp=int(fp.sum()), fn=int(((y == 1)&(pred == 0)).sum()),
                            threshold=meta['threshold'], threshold_source=meta['source'],
                            false_alarm_segments=len(lengths), false_alarm_points_ge_6h=sum(n for n in lengths if n >= 18),
                            false_alarm_length_median=float(np.median(lengths)) if lengths else None,
                            false_alarm_length_p90=float(np.quantile(lengths,.9)) if lengths else None,
                            false_alarm_length_max=max(lengths) if lengths else 0,
                            fit_positives=row['fit_positives'], model=row['model'], start=row['start'], end=row['end']))
        for start, end in segments(y):
            hits = np.flatnonzero(pred[start:end])
            w = df.event_weight.to_numpy()[start:end]
            events.append(dict(device=device, start=df.timestamp.iloc[start], end=df.timestamp.iloc[end-1],
                               points=end-start, missed_event=not bool(len(hits)),
                               first_hit_delay_hours=float(hits[0]/3) if len(hits) else None,
                               recall_3h=float(pred[start:min(end,start+9)].mean()),
                               recall_6h=float(pred[start:min(end,start+18)].mean()),
                               weighted_recall=float(w @ pred[start:end]/w.sum()),
                               weighted_hit=float(w @ pred[start:end]), weighted_total=float(w.sum())))
    dest = ROOT/name
    dest.mkdir(parents=True, exist_ok=True)
    prefix = 'forward_fold1' if fold == 1 else 'diagnostic_fold2'
    pd.DataFrame(devices).to_csv(dest/f'{prefix}_devices.csv',index=False)
    pd.DataFrame(events).to_csv(dest/f'{prefix}_events.csv',index=False)
    pd.DataFrame(false_alarms, columns=['device','start','end','points','hours']).to_csv(dest/f'{prefix}_false_alarm_segments.csv',index=False)
    result = dict(aggregate=aggregate(parts), thresholds=thresholds,
                  events=len(events), missed_events=sum(e['missed_event'] for e in events),
                  macro_recall_3h=float(np.mean([e['recall_3h'] for e in events])) if events else None,
                  macro_recall_6h=float(np.mean([e['recall_6h'] for e in events])) if events else None)
    write_json(dest/f'{prefix}.json',result)
    return result


def verify():
    hashes=json.loads(Path('runs/exp04_protocol/baseline_hashes.json').read_text())
    assert all(sha256(p)==h for p,h in hashes.items()), 'Baseline was modified'
    base=json.loads((RUNS['baseline']/'manifest.json').read_text())
    base_metrics=json.loads((RUNS['baseline']/'metrics.json').read_text())
    control=Path('runs/exp04_baseline_control')
    for f in (RUNS['baseline']/'validation').glob('*.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(f),pd.read_csv(control/'validation'/f.name),check_exact=True)
    checks=[]
    for name,run in RUNS.items():
        manifest=json.loads((run/'manifest.json').read_text())
        metrics=json.loads((run/'metrics.json').read_text())
        for key in ('rounds','threads','seed','gap','fold_starts'):
            assert manifest['config'][key]==base['config'][key], (name,key)
        for d,meta in manifest['models'].items():
            assert meta['source_sha256']==base['models'][d]['source_sha256']
            assert meta['features']==base['models'][d]['features']
            for i in range(3):
                a=pd.read_csv(RUNS['baseline']/'validation'/f'{d}_fold{i}.csv')
                b=pd.read_csv(run/'validation'/f'{d}_fold{i}.csv')
                pd.testing.assert_frame_equal(a[['timestamp','label','event_weight']],b[['timestamp','label','event_weight']],check_exact=True)
        keys=('device','fold','fit_rows','fit_positives','fit_end','start','end','rows','positives','model')
        assert [[r[k] for k in keys] for r in metrics['folds']]==[[r[k] for k in keys] for r in base_metrics['folds']]
        checks.append(name)
    write_json(ROOT/'verification.json',dict(baseline_files_unchanged=len(hashes), default_validation_identical=True,
                                            identical_data_features_splits=checks))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['select','diagnose'])
    args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.stage=='select':
        if (ROOT/'selection.json').exists():
            raise ValueError('Selection already frozen')
        verify()
        results={n:evaluate(n,1,thresholds_from_fold0(r)) for n,r in RUNS.items()}
        selected={m:max([f'{m}_c2',f'{m}_c4'],key=lambda n:results[n]['aggregate']['score']) for m in ('class','event')}
        write_json(ROOT/'selection.json',dict(rule='fold0 calibration -> fold1 score; tie c2; no fold2 selection',selected=selected,
                                             scores={n:r['aggregate'] for n,r in results.items()}))
        print((ROOT/'selection.json').read_text())
    else:
        selection=json.loads((ROOT/'selection.json').read_text())
        results={}
        for name,run in RUNS.items():
            report=json.loads((run/'metrics.json').read_text())
            thresholds={d:dict(threshold=m['threshold'],source=m['threshold_source']) for d,m in report['devices'].items()}
            results[name]=evaluate(name,2,thresholds)
        write_json(ROOT/'results.json',dict(selection=selection,diagnostic=results))
        print(json.dumps({n:r['aggregate'] for n,r in results.items()},indent=2))


if __name__=='__main__':
    main()
