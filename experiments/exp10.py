"""Frozen EXP10 four-cell ablation; exclusively reuses seed42 probability caches."""
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from baseline import sha256, write_json, event_safe_boundary
from metrics import aggregate, event_weights
from experiments.exp10_thresholds import choose_threshold
from experiments.exp10_postprocessing import smooth_probabilities
from experiments.diagnostics import diagnose, segments


def read(p): return json.loads(p.read_text())
def hashes(p): return {str(f.relative_to(p)):sha256(f) for f in p.rglob('*') if f.is_file()}
def block(f): return dict(y=f.label.to_numpy(), p=f.probability.to_numpy())
def transform(f, c):
    f=f.copy()
    if c in ['C2','C3']:
        f['probability']=smooth_probabilities(f.probability.to_numpy(),2)[0]
    return f

def thresholds(locals_, pool, eligible, c, qualification=None):
    shared=choose_threshold(pool)
    den=dict(global_rows=sum(len(b['y']) for b in pool),global_weight=sum(float(event_weights(b['y']).sum()) for b in pool))
    result={}
    for d,bs in locals_.items():
        events=sum(len(segments(b['y'])) for b in bs)
        enough=qualification[d] if qualification is not None else eligible[d] and events>=2 and any((b['y']==0).any() for b in bs)
        kwargs=den if c in ['C1','C3'] and den['global_weight']>0 else {}
        t=choose_threshold(bs,**kwargs) if enough else shared
        result[d]=dict(threshold=t,calibration_events=events,device_threshold_eligible=bool(enough),
                       threshold_source='device_calibration' if enough else 'pooled_calibration' if den['global_weight']>0 else 'default_0.5')
    return result,den

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='runs/exp10_threshold_smoothing')
    args=ap.parse_args(); out=Path(args.output)
    if out.exists() and any(out.iterdir()): ap.error('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    base=ROOT/'runs/baseline_v1'; cache=ROOT/'runs/exp09_forward_protocol'
    before=hashes(base); cache_before=hashes(cache)
    protocol=dict(experiment='EXP10',seed=42,groups={'C0':'device objective, raw','C1':'global objective, raw','C2':'device objective, causal mean2','C3':'global objective, causal mean2'},
        cold_start='Each calibration/evaluation block separately; first point unchanged',
        eligibility='Original >=2 eligible calibration events and normal observations; otherwise pooled/default 0.5',
        monthly_pool='EXP09 eligible prefixes before earliest evaluation, complete events; all four local searches use these same prefixes, retaining original eligibility',
        protocol_amendment='Full local EXP09 calibration may extend beyond shared cutoff. Use common legal prefixes for every cell; separately report C0 vs archived EXP09 differences.',
        primary_unit='Nonduplicate positive device-month task; also report monthly aggregate means',
        stop='C3 early task mean <= C0 OR <= max(C1,C2)',
        gates=read(cache/'protocol.json')['future_candidate_gates'],already_viewed_data=True,
        training_count=0,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(out/'protocol.json',protocol)
    evidence=out/'evidence'; evidence.mkdir()
    for f in ['experiments/exp10.py','experiments/exp10_thresholds.py','experiments/exp10_postprocessing.py','experiments/diagnostics.py','metrics.py','baseline.py','下一轮实验计划_EXP09-EXP11.md']:
        dest=evidence/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    (evidence/'git_diff.patch').write_text(subprocess.check_output(['git','diff','HEAD'],text=True))
    write_json(evidence/'parent_commits.json',{b:subprocess.check_output(['git','rev-parse',b],text=True).strip() for b in ['baseline_v1_exp01','baseline_v1_exp07','baseline_v1_exp09']})
    for rel,h in read(cache/'cache_hashes.json').items(): assert sha256(cache/'tasks'/rel)==h,rel
    for rel,h in read(cache/'data_hashes.json').items(): assert sha256(ROOT/rel)==h,rel
    write_json(out/'input_hashes.json',dict(baseline=before,exp09=cache_before))
    # Stage one: reproduce three archived parent results before monthly analysis.
    original=read(base/'metrics.json'); devices=read(base/'manifest.json')['devices']
    eligible={(r['device'],r['fold']):r['model']=='lightgbm' for r in original['folds']}
    raw={(d,i):pd.read_csv(base/'validation'/f'{d}_fold{i}.csv') for d in devices for i in range(3)}
    historical=[]
    for fold in [1,2]:
        for c in ['C0','C1','C2','C3']:
            locals_={d:[block(transform(raw[d,i],c)) for i in range(fold) if eligible[d,i]] for d in devices}
            pool=[b for bs in locals_.values() for b in bs]
            ts,den=thresholds(locals_,pool,{d:True for d in devices},c); details=[]
            target=out/'historical'/f'fold{fold}'/c; target.mkdir(parents=True)
            for d in devices:
                f=transform(raw[d,fold],c); detail,pred=diagnose(f,ts[d]['threshold']); details.append(detail)
                f['prediction']=pred; f.to_csv(target/f'{d}.csv',index=False)
                write_json(target/f'{d}_metrics.json',dict(**detail,**ts[d]))
                if c in ['C0','C2']:
                    parent='baseline' if c=='C0' else 'b_window_2'
                    archived=pd.read_csv(ROOT/'runs/exp07'/parent/f'fold{fold}'/f'{d}_fold{fold}.csv')
                    np.testing.assert_array_equal(pred,archived.prediction)
                if c=='C1':
                    archived=pd.read_csv(ROOT/'runs/exp08_combination_recheck/exp01'/f'fold{fold}'/f'{d}.csv')
                    np.testing.assert_array_equal(pred,archived.prediction)
            historical.append(dict(fold=fold,group=c,**aggregate(details)))
    pd.DataFrame(historical).to_csv(out/'historical_scores.csv',index=False)
    # Stage two: fixed four cells on every available EXP09 task.
    manifest=read(cache/'split_manifest.json')
    tasks=[t for v in manifest.values() for t in v['tasks'] if t['status']=='available']
    baseline_differences=[]; rows=[]; events=[]; alarms=[]; monthly=[]; frames={c:[] for c in protocol['groups']}; audits=[]
    for month in sorted({t['month'] for t in tasks}):
        mts=[t for t in tasks if t['month']==month]; cutoff=min(t['eval_first_timestamp'] for t in mts)
        for c in protocol['groups']:
            locals_={}; pool=[]; elig={}; qualification={}
            for t in mts:
                d=t['device']; f=transform(pd.read_csv(cache/'tasks'/t['task_id']/'cal_probabilities.csv'),c)
                assert f.timestamp.iloc[-1]<t['eval_first_timestamp']
                n=event_safe_boundary(f.label.to_numpy(),int(pd.DatetimeIndex(pd.to_datetime(f.timestamp)).searchsorted(pd.Timestamp(cutoff))))
                elig[d]=t['eligible_model']; locals_[d]=[block(f.iloc[:n])] if elig[d] else []
                qualification[d]=bool(elig[d] and len(segments(f.label))>=2 and f.label.eq(0).any())
                if elig[d] and n: pool.append(block(f.iloc[:n]))
                # Any locally fitted threshold must have its full calibration included in global denominators.
                assert n==0 or f.timestamp.iloc[n-1]<cutoff
                audits.append(dict(group=c,task_id=t['task_id'],cutoff=cutoff,retained_rows=n,eligible=elig[d]))
            ts,den=thresholds(locals_,pool,elig,c,qualification); summaries=[]
            for t in mts:
                d=t['device']; f=transform(pd.read_csv(cache/'tasks'/t['task_id']/'eval_probabilities.csv'),c)
                detail,pred=diagnose(f,ts[d]['threshold']); summaries.append(detail)
                meta=dict(group=c,task_id=t['task_id'],device=d,month=month)
                rows.append(dict(**meta,**ts[d],**{k:v for k,v in detail.items() if k not in ['events','false_alarm_segments']}))
                events.extend(dict(**meta,event_id=f'{d}:{e["start"]}',**e) for e in detail['events'])
                alarms.extend(dict(**meta,**a) for a in detail['false_alarm_segments'])
                f['prediction']=pred; f['device']=d; frames[c].append(f)
                target=out/'monthly'/c/t['task_id']; target.mkdir(parents=True)
                f.to_csv(target/'evaluation.csv',index=False)
                write_json(target/'metrics.json',dict(**detail,**ts[d],denominators=den,**meta))
                if c=='C0':
                    old=pd.read_csv(cache/'tasks'/t['task_id']/'evaluation.csv').prediction.to_numpy()
                    baseline_differences.append(dict(task_id=t['task_id'],changed_predictions=int((pred!=old).sum()),score_delta=detail['score']-read(cache/'tasks'/t['task_id']/'metrics.json')['score'] if detail['score'] is not None else None))
            monthly.append(dict(month=month,group=c,**aggregate(summaries)))
    pd.DataFrame(baseline_differences).to_csv(out/'C0_vs_exp09.csv',index=False)
    r=pd.DataFrame(rows); e=pd.DataFrame(events); m=pd.DataFrame(monthly)
    r.to_csv(out/'task_metrics.csv',index=False); e.to_csv(out/'event_metrics.csv',index=False)
    pd.DataFrame(alarms).to_csv(out/'false_alarm_segments.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'calibration_pool_audit.csv',index=False)
    m.to_csv(out/'monthly_scores.csv',index=False)
    pivot=r.pivot(index=['task_id','month','device'],columns='group',values='score').reset_index()
    for c in ['C1','C2','C3']: pivot[c+'_delta']=pivot[c]-pivot.C0
    pivot['interaction']=pivot.C3-pivot.C1-pivot.C2+pivot.C0
    pivot.to_csv(out/'paired_tasks.csv',index=False)
    summaries={}
    for name,months in read(cache/'protocol.json')['grouping'].items():
        part=pivot[pivot.month.isin(months)]; valid=part.dropna(subset=['C0'])
        summaries[name]=dict(positive_tasks=len(valid),task_means={c:float(valid[c].mean()) for c in protocol['groups']},
            month_means={c:float(m[m.month.isin(months)&m.group.eq(c)].score.mean()) for c in protocol['groups']},
            interaction_mean=float(valid.interaction.mean()), c3_mean_gain=float(valid.C3_delta.mean()),
            c3_nonnegative_fraction=float(valid.C3_delta.ge(0).mean()),c3_worst_delta=float(valid.C3_delta.min()))
    merged=[]
    for c,fs in frames.items():
        allf=pd.concat(fs,ignore_index=True)
        assert not allf.duplicated(['device','timestamp']).any()
        assert not e[e.group.eq(c)].event_id.duplicated().any()
        for d,f in allf.groupby('device'):
            f=f.sort_values('timestamp').reset_index(drop=True)
            active=(f.label.eq(0)&f.prediction.eq(1)).to_numpy(); gaps=pd.to_datetime(f.timestamp).diff().ne(pd.Timedelta(minutes=20)).to_numpy(); a=None
            for i in range(len(f)+1):
                if a is not None and (i==len(f) or not active[i] or gaps[i]):
                    merged.append(dict(group=c,device=d,start=f.timestamp.iloc[a],end=f.timestamp.iloc[i-1],rows=i-a,minutes=(i-a)*20)); a=None
                if i<len(f) and active[i] and a is None: a=i
    ma=pd.DataFrame(merged); ma.to_csv(out/'deduplicated_false_alarm_segments.csv',index=False)
    longfp={c:int(ma[ma.group.eq(c)&ma.rows.ge(18)].rows.sum()) for c in protocol['groups']}
    ep=e.pivot(index=['event_id','month','device'],columns='group',values='missed_event')
    newmiss=int((ep.C3 & ~ep.C0).sum())
    means=summaries['early']['task_means']; stop=means['C3']<=means['C0'] or means['C3']<=max(means['C1'],means['C2'])
    decision=dict(status='stop_combination' if stop else 'requires_seed_confirmation',adopted=[],retain_baseline=True,
        reason='Predeclared early stopping rule triggered' if stop else 'Early stopping not triggered; additional gates and seed checks required',
        summaries=summaries,new_missed_events_C3=newmiss,long_false_positive_points=longfp,
        seeds_43_44_executed=False,candidate_package_created=False)
    write_json(out/'decision.json',decision)
    write_json(out/'results.json',dict(historical=historical,groups=summaries,aggregate={c:aggregate(r[r.group.eq(c)].to_dict('records')) for c in protocol['groups']}))
    assert hashes(base)==before and hashes(cache)==cache_before
    write_json(out/'verification.json',dict(passed=True,parent_prediction_reproductions=36,exp09_C0_tasks_compared=len(tasks),exp09_C0_changed_predictions=sum(x['changed_predictions'] for x in baseline_differences),data_and_cache_hashes_verified=True,baseline_and_exp09_unchanged=True,unique_rows_and_events=True,local_global_pool_inclusion=True))
    write_json(out/'resources.json',dict(training_count=0,training_seconds=0,elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024)))
    write_json(out/'source_hashes.json',hashes(evidence))
    print(json.dumps(decision,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
