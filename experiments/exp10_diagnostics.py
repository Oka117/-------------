"""Post-run paired event and fixed-weight leave-one-out diagnostics; no selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from metrics import score
from baseline import write_json

def main():
    out=Path(sys.argv[1]); e=pd.read_csv(out/'event_metrics.csv')
    a=e[e.group.eq('C0')].set_index('event_id'); b=e[e.group.eq('C3')].set_index('event_id')
    paired=a[['device','month','start','end','rows']].copy()
    for col in ['missed_event','delay_minutes','recall_3h','recall_6h','weighted_recall']:
        paired[col+'_C0']=a[col]; paired[col+'_C3']=b[col]
    paired.to_csv(out/'paired_events.csv')
    f={c:pd.concat([pd.read_csv(p).assign(nominal_month=p.parent.name[-7:]) for p in sorted((out/'monthly'/c).glob('*/evaluation.csv'))],ignore_index=True) for c in ['C0','C3']}
    pd.testing.assert_frame_equal(f['C0'][['device','timestamp','label','event_weight']],f['C3'][['device','timestamp','label','event_weight']])
    rows=[]; contributions=[]
    for phase in ['early','all']:
        x=f['C0']; scope=x.nominal_month.le('2024-06') if phase=='early' else pd.Series(True,index=x.index)
        n=int(scope.sum()); w=float(x.loc[scope,'event_weight'].sum())
        gains=50*((x.label==f['C3'].prediction).astype(int)-(x.label==x.prediction).astype(int))/n+50*x.event_weight*(f['C3'].prediction-x.prediction)/w
        masks=[('device',d,x.device.eq(d)) for d in x.device.unique()]
        masks += [('event',eid,x.device.eq(v.device)&x.timestamp.ge(v.start)&x.timestamp.le(v.end)) for eid,v in a.iterrows()]
        for kind,key,remove in masks:
            if not (scope&remove).any(): continue
            keep=scope&~remove
            scores={c:score(ff.loc[keep,'label'],ff.loc[keep,'prediction'],ff.loc[keep,'event_weight'])['score'] for c,ff in f.items()}
            rows.append(dict(phase=phase,excluded_kind=kind,excluded=key,C0=scores['C0'],C3=scores['C3'],delta=scores['C3']-scores['C0']))
            contributions.append(dict(phase=phase,kind=kind,key=key,score_contribution=float(gains[scope&remove].sum()),net_gain=float(gains[scope].sum())))
    pd.DataFrame(rows).to_csv(out/'leave_one_out.csv',index=False)
    con=pd.DataFrame(contributions); con['over_half_net_gain']=(con.score_contribution>con.net_gain*.5)&con.net_gain.gt(0)
    con.to_csv(out/'gain_concentration.csv',index=False)
    write_json(out/'diagnostic_summary.json',dict(concentrated=bool(con.over_half_net_gain.any()),
        concentrated_sources=con[con.over_half_net_gain].to_dict('records'),
        events_worse_3h=int((paired.recall_3h_C3<paired.recall_3h_C0).sum()),
        events_worse_6h=int((paired.recall_6h_C3<paired.recall_6h_C0).sum()),
        events_delayed=int((paired.delay_minutes_C3>paired.delay_minutes_C0).sum())))
if __name__=='__main__': main()
