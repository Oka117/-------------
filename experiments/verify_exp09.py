"""Read-only independent cache reload checks; write a separate verification artifact."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd
from baseline import read_train,sha256,write_json,probability
from metrics import event_weights


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default='runs/exp09_forward_protocol')
    args=ap.parse_args()
    out=Path(args.run).resolve()
    target=out/'reload_verification.json'
    if target.exists(): ap.error('Verification output already exists')
    started=time.perf_counter()
    split=json.loads((out/'split_manifest.json').read_text())
    checks=[]
    for d,entry in split.items():
        frame,features,y=read_train(ROOT/'data',d)
        x=frame[features]
        for t in entry['tasks']:
            if t['status']!='available': continue
            folder=out/'tasks'/t['task_id']
            manifest=json.loads((folder/'manifest.json').read_text())
            assert features==manifest['features']
            model=lgb.Booster(model_file=str(folder/manifest['path'])) if manifest['path'] else None
            if model is not None: assert sha256(folder/manifest['path'])==manifest['model_sha256']
            for role in ['cal','eval']:
                cached=pd.read_csv(folder/f'{role}_probabilities.csv')
                a,b=t[role+'_start'],t[role+'_end']
                assert cached.timestamp.tolist()==frame.timestamp.iloc[a:b].tolist()
                np.testing.assert_array_equal(cached.row_index,np.arange(a,b))
                np.testing.assert_array_equal(cached.label,y[a:b])
                np.testing.assert_allclose(cached.event_weight,event_weights(y)[a:b],rtol=0,atol=1e-14)
                assert cached.probability.between(0,1).all()
                # All rows, deliberately crossing 137-point boundaries.
                predicted=np.concatenate([probability(model,manifest['constant'],x.iloc[i:min(i+137,b)],4) for i in range(a,b,137)])
                np.testing.assert_allclose(predicted,cached.probability,rtol=0,atol=1e-14)
            assert t['cal_start']-t['fit_end']>=72 and t['cal_end']==t['eval_start']
            assert pd.Timestamp(t['fit_last_timestamp'])<pd.Timestamp(t['cal_first_timestamp'])
            checks.append(dict(task_id=t['task_id'],model_reloaded=True,calibration_and_evaluation_chunk137_equal=True))
    for relative,expected in json.loads((out/'cache_hashes.json').read_text()).items():
        assert sha256(out/'tasks'/relative)==expected
    for relative,expected in json.loads((out/'baseline_hashes.json').read_text()).items():
        assert sha256(ROOT/'runs/baseline_v1'/relative)==expected
    write_json(target,dict(passed=True,tasks=checks,elapsed_seconds=time.perf_counter()-started,
        tolerance=1e-14,script_sha256=sha256(Path(__file__))))
    print(f'Reload and 137-row chunk parity passed for {len(checks)} tasks')


if __name__=='__main__': main()
