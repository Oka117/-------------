"""Check EXP-03 fold alignment, saved model replay, and real train/test feature parity."""
import argparse
import json
import sys
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline import read_train,read_features,probability,sha256,write_json,temporal_folds
from temporal_features import build_features,inference_features
from experiments.run_exp03 import RUNS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',nargs='+',default=list(RUNS))
    args=ap.parse_args()
    results=[]
    t0=time.perf_counter()
    for name in args.runs:
        run=Path('runs')/name
        m=json.loads((run/'manifest.json').read_text())
        for d,meta in m['models'].items():
            train,cols,y=read_train('data',d)
            for suffix,h in meta['source_sha256'].items():
                assert sha256(Path('data')/d/f'{d}_{suffix}.csv')==h
            config=m['config']
            for f,fit_end,start,end in temporal_folds(train,y,config['fold_starts'],config['gap']):
                stored=pd.read_csv(run/'validation'/f'{d}_fold{f}.csv',float_precision='round_trip')
                base=pd.read_csv(Path('runs/baseline_v1/validation')/f'{d}_fold{f}.csv')
                assert stored.timestamp.equals(base.timestamp)
                np.testing.assert_array_equal(stored.label,base.label)
                recipe=json.loads((run/'feature_recipes'/f'{d}_fold{f}.json').read_text())
                # Only prior 72 raw rows are necessary at the validation boundary.
                x=build_features(train[cols].iloc[max(0,start-72):end],recipe).iloc[72:]
                path=run/'fold_models'/f'{d}_fold{f}.txt'
                model=lgb.Booster(model_file=str(path)) if path.exists() else None
                actual=probability(model,float(y[0]) if model is None else None,x,4)
                np.testing.assert_allclose(actual,stored.probability,rtol=1e-10,atol=1e-12)
                np.testing.assert_array_equal(actual>=meta['threshold'],stored.prediction)
                expected_eps=np.maximum(train[recipe['selected']].iloc[:fit_end].std(ddof=0).fillna(0).to_numpy()*1e-3,1e-6)
                np.testing.assert_allclose(list(recipe['epsilon'].values()),expected_eps,rtol=1e-6)
            test,test_cols=read_features(Path('data')/d/f'{d}_test.csv')
            context,_=read_features(run/meta['history_path'])
            np.testing.assert_array_equal(context[cols],train[cols].tail(72))
            assert list(context.timestamp)==list(train.timestamp.tail(72))
            recipe=meta['feature_recipe']
            x=inference_features(test,context,recipe)
            # Independently reconstruct using all training history when contiguous.
            contiguous=pd.Timestamp(test.timestamp.iloc[0])-pd.Timestamp(train.timestamp.iloc[-1])==pd.Timedelta(minutes=20)
            if contiguous:
                full=pd.concat([train[cols],test[cols]],ignore_index=True)
                expected=build_features(full,recipe).iloc[len(train):].reset_index(drop=True)
            else:
                expected=build_features(test[cols],recipe)
            np.testing.assert_allclose(x,expected,rtol=1e-5,atol=1e-6,equal_nan=True)
            model=lgb.Booster(model_file=str(run/meta['path']))
            p=model.predict(expected,num_threads=4)
            stored=pd.read_csv(run/'predictions'/f'{d}_probabilities.csv',float_precision='round_trip')
            labels=pd.read_csv(run/'predictions'/f'{d}_predict.csv')
            assert stored.timestamp.equals(test.timestamp) and labels.timestamp.equals(test.timestamp)
            assert list(labels.columns)==['pump_id','timestamp','label'] and labels.pump_id.eq(d).all()
            np.testing.assert_allclose(p,stored.probability,rtol=1e-10,atol=1e-12)
            np.testing.assert_array_equal(p>=meta['threshold'],labels.label)
            results.append(dict(run=name,device=d,test_rows=len(test),features=len(meta['features']),
                training_test_contiguous=contiguous,fold_prediction_replay=True,training_only_epsilon=True,
                full_history_vs_saved_context=True,test_prediction_replay=True))
            print('PASS',name,d,flush=True)
    hashes=json.loads(Path('runs/exp03_protocol/baseline_hashes.json').read_text())
    for file,h in hashes.items():assert sha256(file)==h,file
    destination=Path('runs/exp03_analysis/verification.json')
    previous=json.loads(destination.read_text()) if destination.exists() else {'checks':[], 'elapsed_seconds':0}
    retained=[r for r in previous['checks'] if r['run'] not in args.runs]
    write_json(destination,dict(passed=True,baseline_unchanged=True,checks=retained+results,
        elapsed_seconds=previous['elapsed_seconds']+time.perf_counter()-t0))


if __name__=='__main__':main()
