"""Independent EXP06A/B: fixed robust PCA then historical-CDF fusion."""
import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import (read_train, temporal_folds, event_safe_boundary,
                      fit_normal_model, normal_error, sha256, write_json)
from metrics import score, aggregate, choose_threshold, event_weights
from normal_model import bounded_error, fit_cdf, fusion_score

ROOT = Path(__file__).resolve().parents[1]


def segments(mask):
    edge = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def diagnostics(y, p, threshold, timestamp):
    pred = (p >= threshold).astype(np.int8)
    result = score(y, pred)
    result.update(fp=int(((y == 0) & (pred == 1)).sum()),
                  fn=int(((y == 1) & (pred == 0)).sum()))
    events = []
    for a, b in segments(y):
        hit = np.flatnonzero(pred[a:b])
        events.append(dict(start=str(timestamp[a]), end=str(timestamp[b-1]), rows=int(b-a),
            missed_event=not len(hit), delay_hours=float(hit[0]/3) if len(hit) else None,
            recall_first_3h=float(pred[a:min(b,a+9)].mean()),
            recall_first_6h=float(pred[a:min(b,a+18)].mean()),
            weighted_recall=score(y[a:b], pred[a:b])['weighted_recall']))
    false_runs = [dict(start=str(timestamp[a]), end=str(timestamp[b-1]),
                      rows=int(b-a), hours=float((b-a)/3))
                  for a,b in segments((y == 0) & (pred == 1))]
    result.update(events=events, false_alarm_runs=false_runs,
                  false_alarm_run_count=len(false_runs),
                  long_false_alarm_points=sum(r['rows'] for r in false_runs if r['rows'] >= 18),
                  false_alarm_hours_quantiles=(np.quantile([r['hours'] for r in false_runs],
                                               [0,.5,.9,1]).tolist() if false_runs else []))
    return result, pred


def threshold_for(y, p):
    if np.any(y == 1) and np.any(y == 0):
        return choose_threshold([dict(y=y, p=p)]), 'device_historical_calibration'
    normal = p[y == 0]
    if len(normal):
        # Strictly exceed the quantile so tied constant scores cannot alarm all rows.
        return float(np.nextafter(np.quantile(normal, .995, method='higher'), np.inf)), 'normal_q995_no_recall_evidence'
    return float(np.nextafter(1., 2.)), 'no_normal_calibration_no_alarm'


def mappings(block, stop=None):
    sl = slice(None, stop)
    normal = block['y'][sl] == 0
    available = bool(block['eligible']) and len(np.unique(block['supervised'][sl][normal])) > 1
    return dict(normal_cdf=fit_cdf(block['error'][sl][normal]),
                supervised_cdf=fit_cdf(block['supervised'][sl][normal]),
                supervised_available=np.array(available))


def evaluate(blocks, fold, alpha, destination, stage):
    destination.mkdir(parents=True, exist_ok=True)
    details = {}
    for device, bs in blocks.items():
        history = bs[0]
        cut = event_safe_boundary(history['y'], len(history['y'])//2)
        if cut < 2 or cut >= len(history['y']):
            raise ValueError('No event-safe historical mapping/calibration split')
        # fold1: first half fold0 maps, second half fold0 thresholds.
        # fold2/final: all fold0 maps, fold1 thresholds.
        maps = mappings(history, cut if fold == 1 else None)
        cal = history if fold == 1 else bs[1]
        sl = slice(cut, None) if fold == 1 else slice(None)
        if alpha is None:
            cp, ep = bounded_error(cal['error']), bounded_error(bs[fold]['error'])
        else:
            cp = fusion_score(cal['error'], cal['supervised'], maps, alpha)
            ep = fusion_score(bs[fold]['error'], bs[fold]['supervised'], maps, alpha)
        threshold, source = threshold_for(cal['y'][sl], cp[sl])
        detail, pred = diagnostics(bs[fold]['y'], ep, threshold, bs[fold]['timestamp'])
        detail.update(threshold=threshold, threshold_source=source,
            calibration_start=str(cal['timestamp'][sl][0]), calibration_end=str(cal['timestamp'][sl][-1]),
            calibration_events=len(segments(cal['y'][sl])),
            mapping_start=str(history['timestamp'][0]),
            mapping_end=str(history['timestamp'][cut-1 if fold == 1 else -1]),
            mapping_normal_rows=len(maps['normal_cdf']),
            mapping_tail_expected_rows=.005*len(maps['normal_cdf']),
            fusion_available=bool(maps['supervised_available']),
            fusion_fallback='normal_model' if alpha is not None and not bool(maps['supervised_available']) else None,
            score_unique_values=len(np.unique(ep)))
        details[device] = detail
        pd.DataFrame(dict(timestamp=bs[fold]['timestamp'], label=bs[fold]['y'],
            score=ep, prediction=pred, event_weight=event_weights(bs[fold]['y']))).to_csv(
                destination/f'{device}_{stage}.csv', index=False)
        if fold == 2:
            np.savez(destination/f'{device}_mapping.npz', **maps)
    result = dict(aggregate=aggregate(list(details.values())), devices=details)
    write_json(destination/f'{stage}_metrics.json', result)
    return result


def baseline_eval(blocks, manifest, destination, folds=(1, 2)):
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    pooled = [dict(y=bs[0]['y'], p=bs[0]['supervised']) for bs in blocks.values() if bs[0]['eligible']]
    shared = choose_threshold(pooled)
    for fold in folds:
        details = {}
        for device, bs in blocks.items():
            early = bs[0]
            enough = early['eligible'] and len(segments(early['y'])) >= 2 and np.any(early['y']==0)
            threshold = (choose_threshold([dict(y=early['y'], p=early['supervised'])]) if enough else shared)
            if fold == 2:
                threshold = manifest['models'][device]['threshold']
            detail, pred = diagnostics(bs[fold]['y'], bs[fold]['supervised'], threshold, bs[fold]['timestamp'])
            detail['threshold'] = threshold
            detail['threshold_source'] = ('device_fold0' if enough else 'pooled_fold0') if fold == 1 else manifest['models'][device]['threshold_source']
            details[device] = detail
            pd.DataFrame(dict(timestamp=bs[fold]['timestamp'], label=bs[fold]['y'], prediction=pred)).to_csv(destination/f'{device}_fold{fold}.csv',index=False)
        results[f'fold{fold}'] = dict(aggregate=aggregate(list(details.values())), devices=details)
    write_json(destination/'metrics.json',results)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, choices=['exp06'])
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--baseline-dir', default='runs/baseline_v1')
    ap.add_argument('--output', default='runs/exp06')
    args = ap.parse_args()
    out, base = Path(args.output), Path(args.baseline_dir)
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory is not empty')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = json.loads((base/'manifest.json').read_text())
    sources = ['normal_model.py', 'baseline.py', 'train.py', 'predict.py', 'metrics.py',
               'experiments/exp06.py', 'package_submission.py', 'requirements.txt']
    hashes = {str(p):sha256(p) for p in base.rglob('*') if p.is_file()}
    protocol = dict(config=vars(args), pca_variance=.95, solver='float64 covariance eigh, deterministic',
        normal_tail_quantile=.995, alphas=[.25,.5,.75], seed=42,
        mapping='fold0 first event-safe half for fold1; entire fold0 for fold2/final',
        thresholds='remaining fold0 for fold1; fold1 for fold2/final',
        selection='choose B by fold1 only; accept only if beats A and rolling baseline',
        fold2='previously seen development diagnostic; never used to choose alpha',
        baseline='fixed baseline_v1 final diagnostic; fold0 original rule for rolling fold1',
        fusion_fallback='A where fold0 mapping has no eligible variable supervised scores',
        p412b='one labeled event: no independent supervised fault training and fault evaluation',
        final_scale='history-trained score maps reused for full refit; scale transfer is not guaranteed',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,lightgbm=lgb.__version__),
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        baseline_hashes=hashes)
    write_json(out/'protocol.json', protocol)
    for name in sources:
        target=out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    (out/'changes.patch').write_text(subprocess.check_output(['git','diff'],text=True))
    all_blocks, resources = {}, []
    for device in manifest['devices']:
        meta = manifest['models'][device]
        for suffix, expected in meta['source_sha256'].items():
            if sha256(Path(args.data_dir)/device/f'{device}_{suffix}.csv') != expected:
                raise ValueError('Baseline data hash mismatch')
        df, cols, y = read_train(args.data_dir, device)
        if cols != meta['features']:
            raise ValueError('Baseline schema mismatch')
        x = df[cols].to_numpy()
        blocks=[]
        for fold, fit_end, start, end in temporal_folds(df,y,manifest['config']['fold_starts'],manifest['config']['gap']):
            t0=time.perf_counter()
            state=fit_normal_model(x[:fit_end],y[:fit_end])
            fit_seconds=time.perf_counter()-t0
            t0=time.perf_counter()
            error=normal_error(state,x[start:end])
            infer_seconds=time.perf_counter()-t0
            original=pd.read_csv(base/'validation'/f'{device}_fold{fold}.csv')
            if not np.array_equal(original.label,y[start:end]) or not np.array_equal(original.timestamp,df.timestamp.iloc[start:end]):
                raise ValueError('Baseline fold alignment mismatch')
            blocks.append(dict(y=y[start:end],error=error,supervised=original.probability.to_numpy(),
                               timestamp=df.timestamp.iloc[start:end].to_numpy(),eligible=np.unique(y[:fit_end]).size==2))
            folder=out/'a'/ 'models'
            folder.mkdir(parents=True,exist_ok=True)
            np.savez(folder/f'{device}_fold{fold}.npz',**state)
            pd.DataFrame(dict(timestamp=blocks[-1]['timestamp'],label=blocks[-1]['y'],
                reconstruction_error=error,supervised_score=blocks[-1]['supervised'])).to_csv(folder/f'{device}_fold{fold}_scores.csv',index=False)
            resources.append(dict(device=device,fold=fold,fit_rows=fit_end,fit_normal_rows=int(state['normal_rows']),
                fit_positives=int(y[:fit_end].sum()),validation_positives=int(y[start:end].sum()),
                fit_end=str(df.timestamp.iloc[fit_end-1]),start=str(df.timestamp.iloc[start]),end=str(df.timestamp.iloc[end-1]),
                components=int(state['components'].shape[1]),features=len(cols),retained_variance=float(state['retained_variance']),
                fit_seconds=fit_seconds,inference_seconds=infer_seconds))
            print(f'{device} fold{fold}: PCA k={state["components"].shape[1]}, fit {fit_seconds:.2f}s, predict {infer_seconds:.2f}s',flush=True)
        all_blocks[device]=blocks
        t0=time.perf_counter()
        state=fit_normal_model(x,y)
        resources.append(dict(device=device,fold='full',fit_seconds=time.perf_counter()-t0,
            components=int(state['components'].shape[1]),features=len(cols),fit_normal_rows=int(state['normal_rows'])))
        np.savez(out/'a'/'models'/f'{device}.npz',**state)
    # Finish A before any fusion evaluation. No B weight uses fold2 labels.
    a_dev=evaluate(all_blocks,1,None,out/'a'/'validation','fold1')
    development_control=baseline_eval(all_blocks,manifest,out/'baseline_development',folds=(1,))
    write_json(out/'a'/'completed_development.json',dict(status='A development completed',result=a_dev['aggregate']))
    b_dev={str(alpha):evaluate(all_blocks,1,alpha,out/'b'/f'alpha{alpha}'/'validation','fold1') for alpha in protocol['alphas']}
    best=max(protocol['alphas'],key=lambda a:b_dev[str(a)]['aggregate']['score'])
    write_json(out/'selection.json',dict(alpha=best,selected_by='fold1 only',a=a_dev['aggregate'],
        baseline=development_control['fold1']['aggregate'],
        a_passes_screen=a_dev['aggregate']['score'] > development_control['fold1']['aggregate']['score'],
        b_passes_screen=b_dev[str(best)]['aggregate']['score'] > max(a_dev['aggregate']['score'],development_control['fold1']['aggregate']['score']),
        b={a:r['aggregate'] for a,r in b_dev.items()}))
    a_diag=evaluate(all_blocks,2,None,out/'a'/'validation','fold2')
    b_diag=evaluate(all_blocks,2,best,out/'b'/f'alpha{best}'/'validation','fold2')
    controls=baseline_eval(all_blocks,manifest,out/'baseline_control')
    # Package both controls for reproducible inference, regardless of adoption.
    for kind, dest, result in [('normal_pca',out/'a',a_diag),('score_fusion',out/'b'/f'alpha{best}',b_diag)]:
        export=dict(config=protocol,devices=manifest['devices'],models={})
        (dest/'models').mkdir(exist_ok=True)
        for device in manifest['devices']:
            meta=dict(manifest['models'][device])
            meta.update(kind=kind,normal_path=f'models/{device}.npz',threshold=result['devices'][device]['threshold'],
                        threshold_source=result['devices'][device]['threshold_source'])
            if kind=='score_fusion':
                shutil.copy2(out/'a'/meta['normal_path'],dest/meta['normal_path'])
                if meta['path']:
                    shutil.copy2(base/meta['path'],dest/meta['path'])
                meta.update(alpha=best,mapping_path=f'validation/{device}_mapping.npz')
            else:
                meta.update(path=None,constant=None)
            export['models'][device]=meta
        write_json(dest/'manifest.json',export)
        write_json(dest/'config.json',protocol)
    unchanged=all(sha256(path)==h for path,h in hashes.items())
    if not unchanged:
        raise AssertionError('Baseline artifacts changed')
    results=dict(baseline=controls,a=dict(fold1=a_dev,fold2=a_diag),
                 b=dict(alpha=best,fold1=b_dev[str(best)],fold2=b_diag),
                 resources=resources,elapsed_seconds=time.perf_counter()-started,
                 peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
                 baseline_unchanged=unchanged)
    write_json(out/'results.json',results)
    print(json.dumps({k:results[k] for k in ['elapsed_seconds','peak_rss_mib','baseline_unchanged']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
