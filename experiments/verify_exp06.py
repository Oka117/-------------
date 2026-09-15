"""Verify saved EXP06 models, current-time chunk parity and unchanged baseline."""
import json
import sys
import time
import resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lightgbm as lgb
import numpy as np
import pandas as pd
from baseline import read_features, read_train, sha256, probability, write_json
from normal_model import normal_error, bounded_error, fusion_score


def main():
    out=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/exp06')
    results=json.loads((out/'results.json').read_text())
    alpha=results['b']['alpha']
    records=[]
    for relative in ['a',f'b/alpha{alpha}']:
        folder=out/relative
        manifest=json.loads((folder/'manifest.json').read_text())
        for device in manifest['devices']:
            meta=manifest['models'][device]
            df,cols=read_features(Path('data')/device/f'{device}_test.csv')
            x=df[cols].to_numpy()
            start=time.perf_counter()
            with np.load(folder/meta['normal_path'],allow_pickle=False) as state:
                error=normal_error(state,x)
                chunks=np.concatenate([normal_error(state,x[i:i+137]) for i in range(0,len(x),137)])
            np.testing.assert_allclose(error,chunks,rtol=1e-12,atol=1e-10)
            p=bounded_error(error)
            cp=bounded_error(chunks)
            if meta['kind']=='score_fusion':
                model=lgb.Booster(model_file=str(folder/meta['path'])) if meta['path'] else None
                supervised=probability(model,meta['constant'],df[cols],4)
                with np.load(folder/meta['mapping_path'],allow_pickle=False) as maps:
                    p=fusion_score(error,supervised,maps,meta['alpha'])
                    cp=fusion_score(chunks,supervised,maps,meta['alpha'])
            elapsed=time.perf_counter()-start
            expected=pd.read_csv(folder/'predictions'/f'{device}_predict.csv')
            saved=pd.read_csv(folder/'predictions'/f'{device}_probabilities.csv')
            np.testing.assert_array_equal(expected.timestamp,df.timestamp)
            np.testing.assert_array_equal(expected.pump_id,np.full(len(df),device))
            np.testing.assert_array_equal(expected.label,p>=meta['threshold'])
            np.testing.assert_array_equal(expected.label,cp>=meta['threshold'])
            np.testing.assert_allclose(saved.probability,p,rtol=1e-12,atol=1e-15)
            records.append(dict(run=relative,device=device,rows=len(df),labels_equal=True,
                max_chunk_error=float(np.max(np.abs(error-chunks))),
                verification_seconds_including_chunk_repeat=elapsed))
            if relative == 'a':
                training, features, labels=read_train('data',device)
                for fold in range(3):
                    raw=pd.read_csv(folder/'models'/f'{device}_fold{fold}_scores.csv')
                    positions=pd.Index(training.timestamp).get_indexer(raw.timestamp)
                    assert (positions >= 0).all()
                    with np.load(folder/'models'/f'{device}_fold{fold}.npz',allow_pickle=False) as state:
                        reproduced=normal_error(state,training[features].iloc[positions])
                    np.testing.assert_allclose(raw.reconstruction_error,reproduced,rtol=1e-12,atol=1e-10)
    base=Path('runs/baseline_v1')
    for device in results['baseline']['fold2']['devices']:
        fresh=pd.read_csv(out/'baseline_predictions'/f'{device}_predict.csv')
        old=pd.read_csv(base/'predictions'/f'{device}_predict.csv')
        np.testing.assert_array_equal(fresh.timestamp,old.timestamp)
        np.testing.assert_array_equal(fresh.label,old.label)
    protocol=json.loads((out/'protocol.json').read_text())
    assert all(sha256(p)==h for p,h in protocol['baseline_hashes'].items())
    original=json.loads((base/'metrics.json').read_text())
    assert abs(original['holdout_aggregate']['score']-results['baseline']['fold2']['aggregate']['score'])<1e-10
    write_json(out/'verification.json',dict(records=records,baseline_files_unchanged=True,
        baseline_six_device_labels_unchanged=True,baseline_metric_reproduced=True,
        all_18_fold_model_scores_reproduced=True,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)))
    print('Verified A/B six-device scores, labels, schema, chunk parity and unchanged baseline.')


if __name__=='__main__':
    main()
