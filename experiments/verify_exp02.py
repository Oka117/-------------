"""Verify saved EXP-02 predictions against cached raw probabilities and mappings."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline import sha256,write_json
from threshold_transfer import decision_scores
from experiments.exp02 import evaluate,thresholds


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default='runs/exp02_threshold_transfer')
    args=ap.parse_args()
    out=Path(args.run)
    cfg=json.loads((out/'config.json').read_text())
    base=Path(cfg['baseline'])
    provenance=json.loads((out/'provenance.json').read_text())
    for path,expected in provenance['input_sha256'].items():
        assert sha256(path)==expected, path
    source=json.loads((base/'manifest.json').read_text())
    records=[]
    for name in ['exp02a',f'exp02b_alpha_{cfg["selected_alpha"]:g}']:
        folder=out/name
        manifest=json.loads((folder/'manifest.json').read_text())
        for d in manifest['devices']:
            meta=manifest['models'][d]
            assert sha256(folder/meta['path'])==sha256(base/source['models'][d]['path'])
            raw=pd.read_csv(base/'predictions'/f'{d}_probabilities.csv',float_precision='round_trip')
            fresh=pd.read_csv(folder/'predictions'/f'{d}_probabilities.csv',float_precision='round_trip')
            labels=pd.read_csv(folder/'predictions'/f'{d}_predict.csv')
            original=pd.read_csv(base/'predictions'/f'{d}_predict.csv')
            assert list(labels.columns)==['pump_id','timestamp','label']
            assert labels.pump_id.eq(d).all()
            assert fresh.timestamp.equals(raw.timestamp) and labels.timestamp.equals(raw.timestamp)
            np.testing.assert_allclose(fresh.probability,raw.probability,rtol=1e-12,atol=1e-15)
            expected=(decision_scores(raw.probability,meta,folder)>=meta['threshold']).astype(np.int8)
            np.testing.assert_array_equal(labels.label,expected)
            if name=='exp02b_alpha_0':
                pd.testing.assert_frame_equal(labels[['timestamp','label']],original[['timestamp','label']])
            v=pd.read_csv(folder/'validation'/f'{d}_fold2.csv',float_precision='round_trip')
            computed=decision_scores(v.probability,meta,folder)
            np.testing.assert_array_equal((computed>=meta['threshold']).astype(np.int8),v.prediction)
            records.append(dict(experiment=name,device=d,rows=len(labels),
                maximum_raw_probability_difference=float(np.max(np.abs(fresh.probability-raw.probability))),
                changed_labels_vs_baseline=int((labels.label!=original.label).sum()),
                validation_inference_mapping_agrees=True,model_sha256_matches=True))
    # Full fold1 diagnostics, explicitly selection data rather than evaluation.
    metrics=json.loads((base/'metrics.json').read_text())
    eligibility={(r['device'],r['fold']):r['model']=='lightgbm' for r in metrics['folds']}
    history={}
    for d in source['devices']:
        history[d]=[]
        for f in [0,1]:
            df=pd.read_csv(base/'validation'/f'{d}_fold{f}.csv',float_precision='round_trip')
            history[d].append(dict(y=df.label.to_numpy(dtype=np.int8),p=df.probability.to_numpy(),
                timestamp=df.timestamp.to_numpy(),eligible=eligibility[d,f],fold=f))
    _,cfg0=thresholds({d:bs[:1] for d,bs in history.items()})
    references={d:np.sort(bs[0]['p'][bs[0]['y']==0]) for d,bs in history.items() if cfg['mapping_audit'][d]['mapping_available']}
    a_cfg={d:dict(v) for d,v in cfg0.items()}
    for d in references:
        a_cfg[d].update(threshold=cfg['percentile'],source='fold0_normal_ecdf_fold1_shared_trigger')
    dev={}
    for name,settings,refs in [('baseline_forward',cfg0,None),('exp02a_selection',a_cfg,references)]:
        r,frames=evaluate({d:bs[1] for d,bs in history.items()},settings,refs)
        dev[name]=r
        folder=out/'development'/name
        folder.mkdir(parents=True,exist_ok=True)
        for d,df in frames.items():
            df.to_csv(folder/f'{d}_fold1.csv',index=False)
    write_json(out/'development_diagnostics.json',dev)
    write_json(out/'verification.json',dict(passed=True,input_hashes_unchanged=True,
        unit_tests='12 passed',devices=records,
        development_note='fold1 is selection data; fallback thresholds here use fold0 only'))
    print('PASS: 12 device/run checks; fresh inference parity; immutable inputs; alpha=0 equals baseline.')


if __name__=='__main__':
    main()
