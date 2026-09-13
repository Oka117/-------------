"""Independent EXP-02A/B on immutable baseline probabilities; never train models."""
import argparse
import copy
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline import sha256, write_json
from metrics import aggregate, choose_threshold, event_weights, score
from threshold_transfer import normal_percentile, shrink_threshold


def segments(y):
    edge = np.diff(np.r_[0, np.asarray(y, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def thresholds(blocks):
    pooled = [b for bs in blocks.values() for b in bs if b['eligible']]
    shared = choose_threshold(pooled)
    result = {}
    for d, bs in blocks.items():
        valid = [b for b in bs if b['eligible']]
        events = sum(len(segments(b['y'])) for b in valid)
        enough = events >= 2 and any((b['y'] == 0).any() for b in valid)
        result[d] = dict(threshold=choose_threshold(valid) if enough else shared,
                         source='device_calibration' if enough else 'pooled_calibration',
                         eligible=enough, events=events)
    return shared, result


def diagnostics(b, pred, device, fold, threshold, source):
    y = b['y']
    s = score(y, pred)
    s.update(device=device, fold=fold, threshold=threshold, threshold_source=source,
             fp=int(((y == 0) & (pred == 1)).sum()), fn=int(((y == 1) & (pred == 0)).sum()))
    events = []
    for start, end in segments(y):
        hit = np.flatnonzero(pred[start:end])
        w = event_weights(y[start:end])
        events.append(dict(device=device, fold=fold, start=b['timestamp'][start], end=b['timestamp'][end-1],
            points=int(end-start), missed_event=not len(hit),
            first_hit_delay_hours=float(hit[0]/3) if len(hit) else None,
            first_3h_recall=float(pred[start:min(end,start+9)].mean()),
            first_6h_recall=float(pred[start:min(end,start+18)].mean()),
            weighted_recall=float(w[pred[start:end] == 1].sum()/w.sum())))
    fps = [dict(device=device, fold=fold, start=b['timestamp'][a], end=b['timestamp'][z-1],
                points=int(z-a), duration_hours=float((z-a)/3)) for a,z in segments((y == 0) & (pred == 1))]
    lengths = [v['points'] for v in fps]
    s.update(false_alarm_segments=len(fps), long_false_alarm_points=sum(n for n in lengths if n >= 18),
             false_alarm_duration_hours_quantiles=np.quantile(np.array(lengths)/3,[0,.5,.9,1]).tolist() if lengths else [])
    return s, events, fps


def evaluate(blocks, settings, references=None):
    parts, events, fps, frames = [], [], [], {}
    for d, b in blocks.items():
        cfg = settings[d]
        p = normal_percentile(b['p'], references[d]) if references is not None and d in references else b['p']
        pred = (p >= cfg['threshold']).astype(np.int8)
        s, ev, fp = diagnostics(b, pred, d, b['fold'], cfg['threshold'], cfg['source'])
        parts.append(s); events.extend(ev); fps.extend(fp)
        frames[d] = pd.DataFrame(dict(timestamp=b['timestamp'], label=b['y'], probability=b['p'],
                                     decision_score=p, prediction=pred, event_weight=event_weights(b['y'])))
    return dict(aggregate=aggregate(parts), devices=parts, events=events, false_alarm_segments=fps), frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', default='runs/baseline_v1')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--output', default='runs/exp02_threshold_transfer')
    args = ap.parse_args()
    started = time.perf_counter()
    base, out = Path(args.baseline), Path(args.output)
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory is not empty')
    manifest = json.loads((base/'manifest.json').read_text())
    report = json.loads((base/'metrics.json').read_text())
    devices = manifest['devices']
    eligible = {(r['device'],r['fold']):r['model']=='lightgbm' for r in report['folds']}
    checksums = {}
    def load(d, f):
        path = base/'validation'/f'{d}_fold{f}.csv'
        checksums[str(path)] = sha256(path)
        df = pd.read_csv(path, float_precision='round_trip')
        return dict(y=df.label.to_numpy(dtype=np.int8), p=df.probability.to_numpy(),
                    timestamp=df.timestamp.to_numpy(), eligible=eligible[d,f], fold=f)
    # Audit source data and model identity before selecting any parameters.
    for d in devices:
        for suffix, expected in manifest['models'][d]['source_sha256'].items():
            path = Path(args.data_dir)/d/f'{d}_{suffix}.csv'
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f'Source data changed: {path}')
            checksums[str(path)] = actual
        path = base/manifest['models'][d]['path']
        checksums[str(path)] = sha256(path)
    history = {d:[load(d,0),load(d,1)] for d in devices}
    shared0, cfg0 = thresholds({d:bs[:1] for d,bs in history.items()})
    shared, baseline_cfg = thresholds(history)
    for d in devices:
        if baseline_cfg[d]['threshold'] != manifest['models'][d]['threshold']:
            raise AssertionError(f'Baseline threshold replay failed: {d}')
    # A: fold0 normal OOF mapping, fold1 trigger choice. Never fit a mapping
    # from constant-model outputs; unmappable devices retain raw baseline fallback.
    references, audit = {}, {}
    for d, bs in history.items():
        b = bs[0]
        normal = b['p'][b['y']==0]
        valid = b['eligible'] and len(normal)>0
        if valid:
            references[d] = np.sort(normal)
        audit[d] = dict(mapping_available=bool(valid), mapping_fold=0 if valid else None,
                        normal_samples=int(len(normal)) if valid else 0,
                        fallback_reason=None if valid else 'No eligible supervised fold0 normal OOF scores; retain baseline raw threshold',
                        fold0_model_eligible=bs[0]['eligible'],fold1_model_eligible=bs[1]['eligible'])
    selection = [dict(y=history[d][1]['y'],p=normal_percentile(history[d][1]['p'],ref))
                 for d,ref in references.items() if history[d][1]['eligible']]
    if not selection or not any(b['y'].sum() for b in selection):
        raise ValueError('No eligible later anomalies for percentile trigger selection')
    q = choose_threshold(selection)
    a_cfg = copy.deepcopy(baseline_cfg)
    for d, ref in references.items():
        a_cfg[d].update(threshold=q, source='fold0_normal_ecdf_fold1_shared_trigger')
        audit[d].update(shared_percentile=q, reference_tail_points=int((normal_percentile(ref,ref)>=q).sum()),
                        nominal_tail_samples=float(len(ref)*max(0,1-q)))
    # B: choose a single global alpha using only fold0 thresholds -> fold1 score.
    # The alpha=0 candidate exactly preserves baseline threshold endpoints.
    b_selection = []
    for alpha in (0., .5, 1.):
        cfg = copy.deepcopy(cfg0)
        for d in devices:
            if cfg[d]['eligible']:
                cfg[d]['threshold'] = shrink_threshold(cfg[d]['threshold'],shared0,alpha)
        r,_ = evaluate({d:history[d][1] for d in devices},cfg)
        b_selection.append(dict(alpha=alpha,**r))
    chosen = max(b_selection,key=lambda x:x['aggregate']['score'])['alpha']
    settings = {'baseline':baseline_cfg,'exp02a':a_cfg}
    for alpha in (0., .5, 1.):
        cfg = copy.deepcopy(baseline_cfg)
        for d in devices:
            if cfg[d]['eligible']:
                cfg[d].update(threshold=shrink_threshold(cfg[d]['threshold'],shared,alpha),
                              source=f'logit_shrink_alpha_{alpha:g}')
        settings[f'exp02b_alpha_{alpha:g}'] = cfg
    out.mkdir(parents=True, exist_ok=True)
    # Persist the decision before loading fold2: fold2 cannot change selection.
    config = dict(baseline=str(base.resolve()), parent_experiment=None,
        unique_factors=['A: fold0 normal ECDF + fold1 shared percentile','B: logit threshold shrinkage only'],
        percentile=q, selected_alpha=chosen, alpha_tie_break='first: 0, 0.5, 1',
        mapping_audit=audit, settings=settings, b_selection=b_selection,
        evaluation_status='fold2 is previously inspected development diagnostic data',
        no_training=True, score_formula='unchanged baseline metrics.py',
        mapping_policy='No supplementary model training; P202A/P310A/P412B retain raw baseline fallback',
        time_protocol='A: fold0 mapping -> fold1 trigger -> fold2 diagnosis; B: fold0 thresholds -> fold1 alpha -> fold0+1 thresholds -> fold2 diagnosis')
    write_json(out/'config.json',config)
    print(f'Frozen selection: percentile={q:.9g}, alpha={chosen:g}; mapped={list(references)}',flush=True)
    holdout = {d:load(d,2) for d in devices}
    results = {}
    for name,cfg in settings.items():
        r,frames = evaluate(holdout,cfg,references if name=='exp02a' else None)
        results[name] = r
        folder=out/name
        (folder/'validation').mkdir(parents=True)
        for d,df in frames.items():
            df.to_csv(folder/'validation'/f'{d}_fold2.csv',index=False)
        write_json(folder/'metrics.json',r)
        for key in ['devices','events','false_alarm_segments']:
            pd.DataFrame(r[key]).to_csv(folder/f'{key}.csv',index=False)
        print(name,r['aggregate'],flush=True)
    if abs(results['baseline']['aggregate']['score']-report['holdout_aggregate']['score'])>1e-10:
        raise AssertionError('Baseline score replay mismatch')
    if results['exp02b_alpha_0']['aggregate'] != results['baseline']['aggregate']:
        raise AssertionError('Alpha zero must equal baseline')
    # Save deployable inference artifacts for A and the development-selected B.
    for name in ['exp02a',f'exp02b_alpha_{chosen:g}']:
        folder=out/name
        deployed=copy.deepcopy(manifest)
        deployed['experiment']=dict(name=name,baseline=str(base.resolve()),selected_without_fold2=True)
        (folder/'models').mkdir()
        for d in devices:
            meta=deployed['models'][d]
            shutil.copy2(base/meta['path'],folder/meta['path'])
            meta.update(threshold=settings[name][d]['threshold'],threshold_source=settings[name][d]['source'])
            if name=='exp02a' and d in references:
                meta.update(score_transform='normal_percentile',normal_reference=f'models/{d}_normal.npy')
                np.save(folder/meta['normal_reference'],references[d])
        write_json(folder/'manifest.json',deployed)
        write_json(folder/'config.json',config)
    # Distribution shift is descriptive only. Test labels never used; do not tune.
    drift=[]
    for d,bs in history.items():
        for b in bs+[holdout[d]]:
            for label in [0,1]:
                p=b['p'][b['y']==label]
                drift.append(dict(device=d,source=f'fold{b["fold"]}',class_label=label,rows=len(p),
                    quantiles=np.quantile(p,[.5,.9,.99,.999]).tolist() if len(p) else None,
                    eligible=b['eligible']))
        path=base/'predictions'/f'{d}_probabilities.csv'
        if path.exists():
            df=pd.read_csv(path,float_precision='round_trip')
            drift.append(dict(device=d,source='full_model_test_unlabeled',class_label=None,rows=len(df),
                quantiles=np.quantile(df.probability,[.5,.9,.99,.999]).tolist(),eligible=True))
            checksums[str(path)]=sha256(path)
    write_json(out/'score_scale_diagnostics.json',drift)
    snapshot=out/'source_snapshot'
    snapshot.mkdir()
    for name in ['experiments/exp02.py','threshold_transfer.py','predict.py','baseline.py','metrics.py','train.py','package_submission.py']:
        target=snapshot/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/name,target)
    version=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    diff=subprocess.run(['git','diff'],capture_output=True,text=True,check=True).stdout
    (snapshot/'working_tree.patch').write_text(diff)
    write_json(out/'provenance.json',dict(git_head=version,source_sha256={str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*') if p.is_file()},input_sha256=checksums))
    write_json(out/'results.json',results)
    elapsed=time.perf_counter()-started
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024)
    write_json(out/'resources.json',dict(elapsed_seconds=elapsed,training_seconds=0,model_inference_seconds=0,
        note='Uses cached baseline probabilities; separate fresh inference validation required',peak_rss_mib=rss,
        feature_counts={d:len(manifest['models'][d]['features']) for d in devices}))
    print(f'Completed in {elapsed:.2f}s, peak RSS {rss:.1f} MiB; {out.resolve()}',flush=True)


if __name__=='__main__':
    main()
