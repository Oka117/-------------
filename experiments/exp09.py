"""EXP09: frozen, event-complete forward protocol and same-model probability cache."""
import argparse
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd
from baseline import (read_train, event_safe_boundary, temporal_folds, fit_model,
                      probability, sha256, write_json)
from metrics import aggregate, choose_threshold, event_weights
from experiments.diagnostics import segments, diagnose

DATES = [f'2024-{m:02d}-01' for m in range(3,10)]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def hashes(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def partitions(frame, y):
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.timestamp))
    raw = [int(timestamps.searchsorted(pd.Timestamp(d))) for d in DATES] + [len(y)]
    moved = [event_safe_boundary(y, i) for i in raw]
    unique = sorted(set(moved))
    boundaries = [dict(nominal=d, raw_index=a, index=b, moved_points=a-b,
        merged_into_earlier=(b in moved[:i])) for i,(d,a,b) in enumerate(zip(DATES+['train_end_exclusive'],raw,moved))]
    tasks = []
    for i, month in enumerate(DATES[1:],1):
        start = moved[i]
        task = dict(month=month[:7], status='unavailable', reason=None)
        if start in moved[:i] or start == len(y):
            task['reason'] = 'merged_or_empty_evaluation_boundary'
        else:
            j = unique.index(start)
            if j == 0 or j+1 == len(unique):
                task['reason'] = 'missing_previous_or_next_complete_block'
            else:
                cal_start, end = unique[j-1], unique[j+1]
                fit_end = event_safe_boundary(y,max(0,cal_start-72))
                task.update(fit_start=0,fit_end=fit_end,cal_start=cal_start,cal_end=start,
                    eval_start=start,eval_end=end,gap_points=cal_start-fit_end)
                if fit_end < 100 or start-cal_start < 72:
                    task['reason'] = 'minimum_train_100_or_calibration_72_not_met'
                else:
                    task.update(status='available',reason=None)
        tasks.append(task)
    return boundaries,tasks


def assert_complete(y,a,b):
    assert 0 <= a < b <= len(y)
    for k in [a,b]:
        assert k in [0,len(y)] or not (y[k-1] == y[k] == 1), ('split_event',k)


def calibration_block(frame, eligible, cutoff=None):
    """Exclude an entire crossing event when the common time cutoff bisects it."""
    n = len(frame)
    if cutoff is not None:
        n = int(pd.DatetimeIndex(pd.to_datetime(frame.timestamp)).searchsorted(pd.Timestamp(cutoff)))
        n = event_safe_boundary(frame.label.to_numpy(),n)
    f = frame.iloc[:n]
    return dict(y=f.label.to_numpy(),p=f.probability.to_numpy()), dict(
        eligible=bool(eligible), original_rows=len(frame), retained_rows=n, excluded_rows=len(frame)-n,
        retained_end=str(f.timestamp.iloc[-1]) if n else None,
        events=len(segments(f.label.to_numpy())), included=bool(eligible and n))


def compatibility(base,out,manifest):
    old = json.loads((base/'metrics.json').read_text())
    eligible = {(r['device'],r['fold']):r['model']=='lightgbm' for r in old['folds']}
    blocks = {}
    for d in manifest['devices']:
        labels = pd.read_csv(ROOT/'data'/d/f'{d}_train_label.csv')
        y = labels.label.to_numpy()
        for i,fit_end,a,b in temporal_folds(labels,y,manifest['config']['fold_starts'],72):
            f = pd.read_csv(base/'validation'/f'{d}_fold{i}.csv')
            assert f.timestamp.tolist() == labels.timestamp.iloc[a:b].tolist()
            np.testing.assert_array_equal(f.label,y[a:b])
            np.testing.assert_allclose(f.event_weight,event_weights(y)[a:b],rtol=0,atol=1e-14)
            assert_complete(y,a,b)
            assert a-fit_end >= 72
            blocks[d,i] = f
    pool = [dict(y=f.label.to_numpy(),p=f.probability.to_numpy()) for (d,i),f in blocks.items() if i<2 and eligible[d,i]]
    shared = choose_threshold(pool)
    assert abs(shared-old['pooled_threshold']) < 1e-14
    details = {}
    for d in manifest['devices']:
        local = [dict(y=blocks[d,i].label.to_numpy(),p=blocks[d,i].probability.to_numpy()) for i in [0,1] if eligible[d,i]]
        enough = sum(len(segments(b['y'])) for b in local)>=2 and any((b['y']==0).any() for b in local)
        t = choose_threshold(local) if enough else shared
        assert abs(t-old['devices'][d]['threshold']) < 1e-14
        for i in range(3):
            detail,pred = diagnose(blocks[d,i],t)
            np.testing.assert_array_equal(pred,blocks[d,i].prediction)
        details[d] = detail
    result = aggregate(list(details.values()))
    assert abs(result['score']-old['holdout_aggregate']['score']) < 1e-10
    write_json(out/'compatibility.json',dict(passed=True,cached_blocks_checked=18,aggregate=result,
        note='Archived three-block baseline reproduced from immutable cached probabilities; no historical retraining. Calibration-block metrics use the historical combined calibration rule and are not forward validation.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='runs/exp09_forward_protocol')
    args=ap.parse_args()
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory is not empty; choose a new directory')
    out.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    base=ROOT/'runs/baseline_v1'
    before=hashes(base)
    manifest=json.loads((base/'manifest.json').read_text())
    devices=manifest['devices']
    protocol=dict(experiment='EXP09',status='frozen_before_training',seed=42,rounds=300,threads=4,gap_points=72,
        feature_mode='all raw current-time feature_*; baseline hyperparameters',
        candidate_boundaries=DATES+['train_end_exclusive'],evaluation_months=[d[:7] for d in DATES[1:]],
        minimum_train_rows=100,minimum_calibration_rows=72,
        minimum_length_rationale='Fixed engineering availability limits, not optimized: baseline leaf minimum 100; one-day calibration 72. Single-class history still uses baseline constant.',
        march='First complete calibration block; no preceding block among the declared month boundaries, so no March evaluation task.',
        grouping={'early':['2024-04','2024-05','2024-06'],'stability':['2024-07','2024-08'],'late':['2024-09']},
        model_policy='One fit before calibration with event-safe 72-point gap, same frozen model for calibration and evaluation.',
        local_threshold='Baseline rule: eligible mixed-class model, >=2 calibration events and at least one normal; else common pool.',
        shared_pool='Eligible calibration prefixes strictly before earliest target evaluation; drop whole crossing event. No effective positives => 0.5.',
        deduplication='Disjoint per-device evaluation partitions; unique device+timestamp and full event IDs asserted.',
        long_false_alarm_minimum_points=18,already_viewed_data=True,
        limitations='Diagnostic protocol, not an unseen test; boundaries use full label event inventory only for event isolation, not model fitting or threshold selection.',
        future_candidate_gates=dict(mean_gain_min=0.25,nonnegative_fraction_min=2/3,worst_loss_max=1,
            minimum_positive_tasks=3,new_missed_events_max=0,long_fp_relative_increase_max=0.10,
            seeds=[42,43,44],positive_seed_means_min=2,seed_mean_loss_max=0.5,concentration_flag_fraction=0.5),
        scope='Only EXP09 seed42 baseline; no EXP10/11 candidates or submissions')
    write_json(out/'protocol.json',protocol)
    evidence=out/'evidence'; evidence.mkdir()
    sources=['baseline.py','metrics.py','experiments/diagnostics.py','experiments/exp09.py','tests/test_exp09.py',
             'baselinev1_独立实验计划.md','下一轮实验计划_EXP09-EXP11.md']
    for name in sources:
        dest=evidence/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    (evidence/'git.diff').write_text(git('diff','HEAD'))
    (evidence/'git_status.txt').write_text(git('status','--short'))
    write_json(out/'provenance.json',dict(commit=git('rev-parse','HEAD').strip(),branch=git('branch','--show-current').strip(),
        command=[sys.executable,*sys.argv],source_sha256={s:sha256(ROOT/s) for s in sources},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__)))
    data_hashes={}
    for d in devices:
        for suffix,expected in manifest['models'][d]['source_sha256'].items():
            path=ROOT/'data'/d/f'{d}_{suffix}.csv'
            actual=sha256(path); assert actual==expected
            data_hashes[str(path.relative_to(ROOT))]=actual
    write_json(out/'data_hashes.json',data_hashes)
    compatibility(base,out,manifest)
    inventory=[]; split={}; tasklist=[]
    for d in devices:
        labels=pd.read_csv(ROOT/'data'/d/f'{d}_train_label.csv')
        y=labels.label.to_numpy()
        boundaries,tasks=partitions(labels,y)
        for a,b in segments(y):
            inventory.append(dict(device=d,event_id=f'{d}:{labels.timestamp.iloc[a]}',start=str(labels.timestamp.iloc[a]),
                end=str(labels.timestamp.iloc[b-1]),start_index=int(a),end_exclusive=int(b),rows=int(b-a)))
        for t in tasks:
            t.update(device=d,task_id=f'{d}_{t["month"]}')
            if t['status']=='available':
                for prefix,a,b in [('fit',0,t['fit_end']),('cal',t['cal_start'],t['cal_end']),('eval',t['eval_start'],t['eval_end'])]:
                    assert_complete(y,a,b)
                    t[prefix+'_first_timestamp']=str(labels.timestamp.iloc[a])
                    t[prefix+'_last_timestamp']=str(labels.timestamp.iloc[b-1])
                    t[prefix+'_rows']=b-a; t[prefix+'_positives']=int(y[a:b].sum())
                    t[prefix+'_events']=len(segments(y[a:b]))
                assert t['fit_end']+72<=t['cal_start']<t['cal_end']==t['eval_start']<t['eval_end']
                t['eligible_model']=np.unique(y[:t['fit_end']]).size==2
            tasklist.append(t)
        split[d]=dict(boundaries=boundaries,tasks=tasks)
    pd.DataFrame(inventory).to_csv(out/'event_inventory.csv',index=False)
    write_json(out/'split_manifest.json',split)
    pd.DataFrame(tasklist).to_csv(out/'task_inventory.csv',index=False)
    available=[t for t in tasklist if t['status']=='available']
    timing=[]; cache={}; total_training=0.; total_predict=0.
    for d in devices:
        df,cols,y=read_train(ROOT/'data',d)
        x=df[cols]; weights=event_weights(y)
        for t in [t for t in available if t['device']==d]:
            folder=out/'tasks'/t['task_id']; folder.mkdir(parents=True)
            start=time.perf_counter()
            model,constant=fit_model(x.iloc[:t['fit_end']],y[:t['fit_end']],300,4,42)
            fit_seconds=time.perf_counter()-start; total_training+=fit_seconds
            if model is not None: model.save_model(str(folder/'model.txt'))
            start=time.perf_counter(); frames={}
            for role in ['cal','eval']:
                a,b=t[role+'_start'],t[role+'_end']
                frames[role]=pd.DataFrame(dict(row_index=np.arange(a,b),timestamp=df.timestamp.iloc[a:b].to_numpy(),
                    label=y[a:b],probability=probability(model,constant,x.iloc[a:b],4),event_weight=weights[a:b]))
                np.testing.assert_allclose(frames[role].event_weight,event_weights(y[a:b]),rtol=0,atol=0)
                frames[role].to_csv(folder/f'{role}_probabilities.csv',index=False)
            predict_seconds=time.perf_counter()-start; total_predict+=predict_seconds
            model_meta=dict(task=t,features=cols,constant=constant,path='model.txt' if model is not None else None,
                model_sha256=sha256(folder/'model.txt') if model is not None else None,
                fit_seconds=fit_seconds,predict_and_cache_seconds=predict_seconds)
            write_json(folder/'manifest.json',model_meta)
            cache[t['task_id']]=frames
            timing.append(dict(task_id=t['task_id'],mixed_class=model is not None,fit_seconds=fit_seconds,predict_and_cache_seconds=predict_seconds))
            print(f'{t["task_id"]}: fit={fit_seconds:.2f}s predict/cache={predict_seconds:.2f}s model={"LGBM" if model is not None else constant}',flush=True)
            if len(timing)==2:
                estimate=dict(measured_tasks=timing.copy(),remaining_tasks=len(available)-2,
                    rough_remaining_seconds=(fit_seconds+predict_seconds+timing[0]['fit_seconds']+timing[0]['predict_and_cache_seconds'])/2*(len(available)-2),
                    caveat='Two-task CPU extrapolation only; later histories and constant models have different costs.')
                write_json(out/'pilot_timing.json',estimate)
                print('PILOT '+json.dumps(estimate),flush=True)
        del df,x,model
    month_results={}; task_results=[]; event_results=[]; alarm_results=[]; evaluation_frames=[]; calibration_audit=[]
    for month in protocol['evaluation_months']:
        tasks=[t for t in available if t['month']==month]
        if not tasks:
            month_results[month]=dict(participating_devices=[],missing_devices=devices,aggregate=None); continue
        cutoff=min(t['eval_first_timestamp'] for t in tasks)
        pool=[]
        for t in tasks:
            block,audit=calibration_block(cache[t['task_id']]['cal'],t['eligible_model'],cutoff)
            audit.update(task_id=t['task_id'],month=month,common_cutoff=cutoff)
            calibration_audit.append(audit)
            if audit['included']:
                assert pd.Timestamp(audit['retained_end']) < pd.Timestamp(cutoff)
                pool.append(block)
        shared=choose_threshold(pool)
        summaries=[]
        for t in tasks:
            frames=cache[t['task_id']]
            local,audit=calibration_block(frames['cal'],t['eligible_model'])
            assert pd.Timestamp(frames['cal'].timestamp.iloc[-1]) < pd.Timestamp(t['eval_first_timestamp'])
            enough=t['eligible_model'] and audit['events']>=2 and (local['y']==0).any()
            has_both=any(b['y'].sum() for b in pool) and any((b['y']==0).any() for b in pool)
            threshold=choose_threshold([local]) if enough else shared
            source='device_calibration' if enough else 'pooled_calibration' if has_both else 'default_0.5_no_valid_pool'
            detail,pred=diagnose(frames['eval'],threshold)
            detail.update(task_id=t['task_id'],device=t['device'],month=month,threshold=threshold,threshold_source=source,
                calibration_events=audit['events'],device_threshold_eligible=bool(enough),mixed_class_model=t['eligible_model'])
            write_json(out/'tasks'/t['task_id']/'metrics.json',detail)
            frame=frames['eval'].copy(); frame['prediction']=pred; frame['device']=t['device']; frame['month']=month
            frame.to_csv(out/'tasks'/t['task_id']/'evaluation.csv',index=False); evaluation_frames.append(frame)
            for e in detail['events']:
                event_results.append(dict(task_id=t['task_id'],device=t['device'],month=month,event_id=f'{t["device"]}:{e["start"]}',**e))
            for a in detail['false_alarm_segments']:
                alarm_results.append(dict(task_id=t['task_id'],device=t['device'],month=month,**a))
            row={k:v for k,v in detail.items() if k not in ['events','false_alarm_segments']}
            task_results.append(row); summaries.append(row)
        month_results[month]=dict(participating_devices=[t['device'] for t in tasks],
            missing_devices=[d for d in devices if d not in [t['device'] for t in tasks]],
            common_pool_cutoff=cutoff,pooled_threshold=shared,aggregate=aggregate(summaries),
            positive_device_task_mean=float(np.mean([s['score'] for s in summaries if s['score'] is not None])) if any(s['score'] is not None for s in summaries) else None)
    allrows=pd.concat(evaluation_frames,ignore_index=True)
    assert not allrows.duplicated(['device','timestamp']).any()
    events=pd.DataFrame(event_results)
    assert not events.event_id.duplicated().any()
    full_events={e['event_id']:e for e in inventory}
    for e in event_results:
        assert e['rows']==full_events[e['event_id']]['rows'] and e['end']==full_events[e['event_id']]['end']
    # Merge adjacent false-alarm runs across monthly task boundaries for the unique-time total.
    merged_alarms=[]
    for d,f in allrows.groupby('device',sort=True):
        f=f.sort_values('timestamp').reset_index(drop=True)
        active=(f.label.eq(0)&f.prediction.eq(1)).to_numpy()
        gaps=pd.to_datetime(f.timestamp).diff().ne(pd.Timedelta(minutes=20)).to_numpy()
        a=None
        for i in range(len(f)+1):
            if a is not None and (i==len(f) or not active[i] or gaps[i]):
                merged_alarms.append(dict(device=d,start=str(f.timestamp.iloc[a]),end=str(f.timestamp.iloc[i-1]),rows=i-a,minutes=(i-a)*20)); a=None
            if i<len(f) and active[i] and a is None: a=i
    pd.DataFrame(task_results).to_csv(out/'task_metrics.csv',index=False)
    events.to_csv(out/'event_metrics.csv',index=False)
    pd.DataFrame(alarm_results).to_csv(out/'false_alarm_segments.csv',index=False)
    pd.DataFrame(merged_alarms).to_csv(out/'deduplicated_false_alarm_segments.csv',index=False)
    pd.DataFrame(calibration_audit).to_csv(out/'calibration_pool_audit.csv',index=False)
    groups={}
    for name,months in protocol['grouping'].items():
        rows=[r for r in task_results if r['month'] in months]
        scores=[month_results[m]['aggregate']['score'] for m in months if month_results[m]['aggregate'] is not None and month_results[m]['aggregate']['score'] is not None]
        groups[name]=dict(aggregate=aggregate(rows),positive_month_count=len(scores),month_mean_score=float(np.mean(scores)) if scores else None,
            positive_device_task_count=sum(r['score'] is not None for r in rows),
            device_task_mean_score=float(np.mean([r['score'] for r in rows if r['score'] is not None])) if any(r['score'] is not None for r in rows) else None)
    write_json(out/'results.json',dict(months=month_results,groups=groups,aggregate=aggregate(task_results),
        available_tasks=len(available),unavailable_tasks=len(tasklist)-len(available),mixed_class_tasks=sum(t['eligible_model'] for t in available),
        unique_evaluated_events=len(events),missed_events=int(events.missed_event.sum()),
        unique_long_false_positive_points=sum(a['rows'] for a in merged_alarms if a['rows']>=18)))
    assert hashes(base)==before
    for path,expected in data_hashes.items(): assert sha256(ROOT/path)==expected
    write_json(out/'baseline_hashes.json',before)
    write_json(out/'verification.json',dict(passed=True,baseline_unchanged=True,data_unchanged=True,
        no_fit_calibration_evaluation_overlap=True,minimum_gap_72=True,full_events_preserved=True,
        common_pool_strictly_before_earliest_evaluation=True,no_duplicate_evaluation_rows=True,no_duplicate_evaluation_events=True,
        models_shared_between_calibration_and_evaluation=True,compatibility_18_cached_blocks_passed=True))
    write_json(out/'resources.json',dict(fit_seconds=total_training,predict_and_cache_seconds=total_predict,
        elapsed_seconds=time.perf_counter()-began,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024),tasks=timing))
    write_json(out/'cache_hashes.json',hashes(out/'tasks'))
    print('DONE '+str(out),flush=True)


if __name__=='__main__': main()
