"""Measure EXP04 inference in fresh processes and verify output alignment."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline import write_json
from exp04 import RUNS, ROOT

wrapper = '''import runpy,sys,time,resource,platform,json
args=sys.argv[1:];output=args[args.index('--output')+1]
sys.argv=['predict.py']+args
start=time.perf_counter();runpy.run_path('predict.py',run_name='__main__')
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with open(output+'/resources.json','w') as f:
 json.dump(dict(elapsed_seconds=time.perf_counter()-start,peak_rss_mib=rss/(1024**2 if platform.system()=='Darwin' else 1024)),f,indent=2)
'''
results={}
for name,run in RUNS.items():
    out=ROOT/'baseline_predictions' if name=='baseline' else run/'predictions'
    subprocess.run([sys.executable,'-c',wrapper,'--model-dir',str(run),'--output',str(out),'--threads','4'],check=True)
    manifest=json.loads((run/'manifest.json').read_text())
    counts={}
    for d in manifest['devices']:
        expected=pd.read_csv(Path('data')/d/f'{d}_test.csv',usecols=['timestamp'])
        actual=pd.read_csv(out/f'{d}_predict.csv')
        prob=pd.read_csv(out/f'{d}_probabilities.csv')
        assert actual.columns.tolist()==['pump_id','timestamp','label']
        assert actual.timestamp.equals(expected.timestamp) and prob.timestamp.equals(expected.timestamp)
        assert actual.pump_id.eq(d).all() and actual.label.isin([0,1]).all()
        assert np.isfinite(prob.probability).all() and prob.probability.between(0,1).all()
        np.testing.assert_array_equal(actual.label,(prob.probability>=manifest['models'][d]['threshold']).astype(int))
        counts[d]=len(actual)
    results[name]=dict(**json.loads((out/'resources.json').read_text()),verified_rows=counts)
write_json(ROOT/'inference_verification.json',results)
