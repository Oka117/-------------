"""EXP08 admission audit: do not combine changes rejected by their parent experiments."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from baseline import sha256, write_json, temporal_folds
from metrics import aggregate, choose_threshold, event_weights, score


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def segments(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def diagnose(frame, threshold):
    y = frame.label.to_numpy()
    pred = (frame.probability.to_numpy() >= threshold).astype(np.int8)
    result = score(y, pred)
    result.update(FP=int(((y == 0) & (pred == 1)).sum()), FN=int(((y == 1) & (pred == 0)).sum()))
    result['events'] = []
    for a, b in segments(y):
        hits = np.flatnonzero(pred[a:b])
        result['events'].append(dict(start=frame.timestamp.iloc[a], end=frame.timestamp.iloc[b-1], rows=int(b-a),
            missed_event=not len(hits), delay_minutes=int(hits[0])*20 if len(hits) else None,
            recall_3h=float(pred[a:min(b,a+9)].mean()), recall_6h=float(pred[a:min(b,a+18)].mean()),
            weighted_recall=float(np.dot(event_weights(y[a:b]), pred[a:b])/event_weights(y[a:b]).sum())))
    result['false_alarm_segments'] = [dict(start=frame.timestamp.iloc[a], end=frame.timestamp.iloc[b-1],
        rows=int(b-a), minutes=int(b-a)*20) for a,b in segments((y == 0) & (pred == 1))]
    result['long_false_positive_points'] = sum(s['rows'] for s in result['false_alarm_segments'] if s['rows'] >= 18)
    return result, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='runs/exp08_combination')
    args = ap.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory is not empty; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = ROOT/'runs/baseline_v1'
    before = {str(p.relative_to(base)): sha256(p) for p in base.rglob('*') if p.is_file()}
    evidence = out/'evidence'
    evidence.mkdir()
    branches = {}
    for i in range(1,8):
        branch = f'baseline_v1_exp{i:02}'
        commit = git('rev-parse', branch).strip()
        report = f'EXP{i:02}_实验结果.md'
        content = git('show', f'{commit}:{report}')
        (evidence/report).write_text(content)
        branches[branch] = dict(commit=commit, report=report, report_sha256=sha256(evidence/report))
    write_json(out/'protocol.json', dict(status='admission_audit',
        parent_decisions='Manual review of the seven archived branch reports; selected candidate does not mean accepted improvement.',
        rule='Only changes accepted by their completed independent stability validation may be combined. Do not relax parent criteria after viewing outcomes.',
        exp01_check='fold0 calibration -> fold1; fold0+fold1 calibration -> already viewed fold2 diagnostic',
        combinations_require_at_least=2, training_seconds=0, baseline_commit=git('rev-parse','HEAD').strip(), branches=branches))
    # Reuse the exact EXP01 threshold implementation from the recorded parent commit.
    source = git('show', f"{branches['baseline_v1_exp01']['commit']}:metrics.py")
    module_file = evidence/'exp01_metrics.py'
    module_file.write_text(source)
    spec = importlib.util.spec_from_file_location('exp01_metrics_snapshot', module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((base/'manifest.json').read_text())
    original = json.loads((base/'metrics.json').read_text())
    assert len(manifest['devices']) == 6
    eligible = {(r['device'],r['fold']): r['model']=='lightgbm' for r in original['folds']}
    data_hashes, blocks = {}, {}
    for d in manifest['devices']:
        for suffix, expected in manifest['models'][d]['source_sha256'].items():
            path = ROOT/'data'/d/f'{d}_{suffix}.csv'
            actual = sha256(path)
            assert actual == expected, path
            data_hashes[str(path.relative_to(ROOT))] = actual
        labels = pd.read_csv(ROOT/'data'/d/f'{d}_train_label.csv')
        ranges = list(temporal_folds(labels, labels.label.to_numpy(), manifest['config']['fold_starts'], manifest['config']['gap']))
        for i, _, a, b in ranges:
            frame = pd.read_csv(base/'validation'/f'{d}_fold{i}.csv')
            assert frame.timestamp.tolist() == labels.timestamp.iloc[a:b].tolist()
            np.testing.assert_array_equal(frame.label, labels.label.iloc[a:b])
            other = pd.read_csv(ROOT/'runs/exp01_global_threshold/validation'/f'{d}_fold{i}.csv')
            pd.testing.assert_frame_equal(frame[['timestamp','label','probability']], other[['timestamp','label','probability']])
            blocks[d,i] = frame
    results = {}
    fold_predictions = {}
    for fold in [1,2]:
        pool = [dict(y=blocks[d,i].label.to_numpy(), p=blocks[d,i].probability.to_numpy())
                for d in manifest['devices'] for i in range(fold) if eligible[d,i]]
        shared = choose_threshold(pool)
        denoms = dict(global_rows=sum(len(b['y']) for b in pool), global_weight=sum(float(event_weights(b['y']).sum()) for b in pool))
        for mode in ['baseline','exp01']:
            target = out/mode/f'fold{fold}'
            target.mkdir(parents=True)
            devices = {}
            for d in manifest['devices']:
                local = [dict(y=blocks[d,i].label.to_numpy(),p=blocks[d,i].probability.to_numpy()) for i in range(fold) if eligible[d,i]]
                events = sum(len(segments(b['y'])) for b in local)
                enough = events >= 2 and any((b['y']==0).any() for b in local)
                threshold = module.choose_threshold(local, **(denoms if mode=='exp01' else {})) if enough else shared
                detail, pred = diagnose(blocks[d,fold], threshold)
                fold_predictions[mode,fold,d] = pred
                detail.update(threshold=threshold, threshold_source='device_calibration' if enough else 'pooled_calibration', calibration_events=events)
                devices[d] = detail
                frame = blocks[d,fold].copy()
                frame['prediction'] = pred
                frame.to_csv(target/f'{d}.csv',index=False)
                if fold == 2:
                    old = base if mode=='baseline' else ROOT/'runs/exp01_global_threshold'
                    np.testing.assert_array_equal(pred, pd.read_csv(old/'validation'/f'{d}_fold2.csv').prediction)
            value = dict(aggregate=aggregate(list(devices.values())), devices=devices, denominators=denoms)
            write_json(target/'metrics.json',value)
            results[f'{mode}_fold{fold}'] = value['aggregate']
    fold1_identical = all(np.array_equal(fold_predictions['baseline',1,d], fold_predictions['exp01',1,d]) for d in manifest['devices'])
    if not fold1_identical:
        raise RuntimeError('EXP01 forward evidence changed: review admission conclusions before proceeding')
    decisions = [
        dict(experiment='EXP01', accepted=False, reason='Earlier fold1 has no prediction or score gain; fold2 gain remains concentrated in one P601B event. Stability evidence insufficient; provisional only.'),
        dict(experiment='EXP02', accepted=False, reason='02A rejected; 02B selects unchanged baseline alpha=0.'),
        dict(experiment='EXP03', accepted=False, reason='Selected 03A loses fold2 across seeds 42/43/44.'),
        dict(experiment='EXP04', accepted=False, reason='Event c2 fails paired seed stability; parent explicitly excludes EXP08 admission.'),
        dict(experiment='EXP05', accepted=False, reason='Parent decision adopted=[]; leaf50 loses later blocks and seed checks.'),
        dict(experiment='EXP06', accepted=False, reason='Both A and selected B fail earlier fold1 comparison.'),
        dict(experiment='EXP07', accepted=False, reason='Both selected candidates fail parent long-false-alarm stability criterion.')]
    write_json(out/'decision.json',dict(status='not_eligible_for_combination',adopted=[],decisions=decisions,
        combination_training_executed=False, removal_ablation_executed=False, candidate_package_created=False,
        reason='No admitted changes. Combination and removal ablation preconditions unmet; retain baselinev1.'))
    # Fresh inference smoke check for the retained baseline; no candidate submission package.
    t0 = time.perf_counter()
    command = [sys.executable,str(ROOT/'predict.py'),'--data-dir',str(ROOT/'data'),'--model-dir',str(base),'--output',str(out/'baseline_predictions')]
    completed = subprocess.run(command,check=True,capture_output=True,text=True)
    (out/'inference.log').write_text(completed.stdout)
    inference_seconds = time.perf_counter()-t0
    checks = {}
    for d in manifest['devices']:
        pred = pd.read_csv(out/'baseline_predictions'/f'{d}_predict.csv')
        old = pd.read_csv(base/'predictions'/f'{d}_predict.csv')
        test = pd.read_csv(ROOT/'data'/d/f'{d}_test.csv',usecols=['timestamp'])
        assert pred.columns.tolist() == ['pump_id','timestamp','label']
        assert pred.pump_id.eq(d).all() and pred.label.isin([0,1]).all()
        pd.testing.assert_series_equal(pred.timestamp,test.timestamp)
        pd.testing.assert_frame_equal(pred[['timestamp','label']],old[['timestamp','label']])
        checks[d] = dict(rows=len(pred), labels_equal_baseline=True, timestamps_equal_test=True)
    after = {str(p.relative_to(base)): sha256(p) for p in base.rglob('*') if p.is_file()}
    assert before == after
    write_json(out/'baseline_hashes.json',before)
    write_json(out/'data_hashes.json',data_hashes)
    write_json(out/'results.json',results)
    write_json(out/'verification.json',dict(passed=True, baseline_unchanged=True, historical_csv_pairs_checked=18,
        fold2_predictions_reproduced=True, exp01_fold1_labels_identical=True, fresh_inference=checks))
    write_json(out/'resources.json',dict(training_seconds=0,inference_seconds=inference_seconds,elapsed_seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024)))
    (evidence/'exp08.py').write_text(Path(__file__).read_text())
    print(json.dumps(results,indent=2))
    print('EXP08 admission audit complete:',out)


if __name__=='__main__':
    main()
