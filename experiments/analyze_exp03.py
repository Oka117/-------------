"""Audit EXP-03, choose a family using forward fold1 only, then diagnose fold2."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline import write_json,sha256
from metrics import aggregate,choose_threshold,event_weights,score
from experiments.run_exp03 import RUNS


def segments(y):
    edges=np.diff(np.r_[0,np.asarray(y,dtype='int8'),0])
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def read_blocks(run, folds):
    metadata=json.loads((run/'metrics.json').read_text())
    eligible={(v['device'],v['fold']):v['model']=='lightgbm' for v in metadata['folds']}
    blocks={}
    for d in metadata['devices']:
        blocks[d]=[]
        for f in folds:
            df=pd.read_csv(run/'validation'/f'{d}_fold{f}.csv',float_precision='round_trip')
            blocks[d].append(dict(y=df.label.to_numpy(dtype='int8'),p=df.probability.to_numpy(),
                timestamp=df.timestamp.to_numpy(),eligible=eligible[d,f],fold=f))
    return metadata,blocks


def calibrate(blocks):
    pooled=[b for bs in blocks.values() for b in bs if b['eligible']]
    shared=choose_threshold(pooled)
    thresholds={}
    for d,bs in blocks.items():
        valid=[b for b in bs if b['eligible']]
        count=sum(len(segments(b['y'])) for b in valid)
        enough=count>=2 and any((b['y']==0).any() for b in valid)
        thresholds[d]=dict(value=choose_threshold(valid) if enough else shared,
                           source='device_calibration' if enough else 'pooled_calibration',events=count)
    return thresholds


def evaluate(blocks,thresholds):
    devices=[];events=[];fps=[]
    for d,bs in blocks.items():
        for b in bs:
            y=b['y'];p=b['p'];t=thresholds[d]['value'];pred=(p>=t).astype('int8')
            s=score(y,pred);n1=int(y.sum());n0=len(y)-n1
            auc=float((pd.Series(p).rank()[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
            s.update(device=d,fold=b['fold'],threshold=t,threshold_source=thresholds[d]['source'],
                     fp=int(((y==0)&(pred==1)).sum()),fn=int(((y==1)&(pred==0)).sum()),auc=auc)
            lengths=[]
            for start,end in segments((y==0)&(pred==1)):
                lengths.append(int(end-start))
                fps.append(dict(device=d,fold=b['fold'],start=b['timestamp'][start],end=b['timestamp'][end-1],points=int(end-start),hours=float((end-start)/3)))
            s.update(false_alarm_segments=len(lengths),long_false_alarm_points=sum(n for n in lengths if n>=18),
                     false_alarm_hours_quantiles=np.quantile(np.array(lengths)/3,[0,.5,.9,1]).tolist() if lengths else [])
            devices.append(s)
            for start,end in segments(y):
                hit=np.flatnonzero(pred[start:end]);w=event_weights(y[start:end])
                events.append(dict(device=d,fold=b['fold'],start=b['timestamp'][start],end=b['timestamp'][end-1],points=int(end-start),
                    missed_event=not len(hit),first_hit_delay_hours=float(hit[0]/3) if len(hit) else None,
                    first_3h_recall=float(pred[start:min(end,start+9)].mean()),first_6h_recall=float(pred[start:min(end,start+18)].mean()),
                    weighted_recall=float(w[pred[start:end]==1].sum()/w.sum())))
    return dict(aggregate=aggregate(devices),devices=devices,events=events,false_alarm_segments=fps)


def main():
    out=Path('runs/exp03_analysis')
    out.mkdir(exist_ok=True)
    names=['baseline_v1',*RUNS]
    forward={};thresholds={}
    for name in names:
        meta,blocks=read_blocks(Path('runs')/name,[0,1])
        cfg=calibrate({d:bs[:1] for d,bs in blocks.items()})
        forward[name]=evaluate({d:bs[1:] for d,bs in blocks.items()},cfg)
        thresholds[name]=calibrate(blocks)
    # Predeclared choice uses only fold0 -> fold1 performance. Baseline wins ties.
    chosen=max(names,key=lambda n:forward[n]['aggregate']['score'])
    write_json(out/'selection.json',dict(protocol='fold0 thresholds -> fold1 family selection; baseline wins ties',
        chosen=chosen,scores={n:r['aggregate'] for n,r in forward.items()},
        second_stage_seeds=[43,44] if chosen!='baseline_v1' else [],
        note='fold2 previously inspected, diagnostic only; no combined feature families'))
    print('FORWARD SELECTION',chosen,{n:r['aggregate']['score'] for n,r in forward.items()},flush=True)
    results={}
    for name in names:
        run=Path('runs')/name
        meta,blocks=read_blocks(run,[2])
        results[name]=evaluate(blocks,thresholds[name])
        if abs(results[name]['aggregate']['score']-meta['holdout_aggregate']['score'])>1e-10:
            raise AssertionError('Saved threshold score mismatch')
        folder=out/name;folder.mkdir(exist_ok=True)
        for split,r in [('forward_fold1',forward[name]),('diagnostic_fold2',results[name])]:
            write_json(folder/f'{split}.json',r)
            for key in ['devices','events','false_alarm_segments']:
                pd.DataFrame(r[key]).to_csv(folder/f'{split}_{key}.csv',index=False)
        print(name,results[name]['aggregate'],flush=True)
    write_json(out/'results.json',results)
    write_json(out/'forward_results.json',forward)


if __name__=='__main__':main()
