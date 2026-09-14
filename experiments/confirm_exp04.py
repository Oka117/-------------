"""Analyze the frozen event_c2 candidate at seeds 43 and 44."""
import json
from pathlib import Path
from exp04 import RUNS, ROOT, evaluate, thresholds_from_fold0
from baseline import write_json

results=[]
for seed in (43,44):
    row={'seed':seed}
    for mode in ('none','event'):
        name=f'{mode}_seed{seed}'
        run=Path('runs/exp04_confirmation')/name
        RUNS[name]=run
        forward=evaluate(name,1,thresholds_from_fold0(run))
        metrics=json.loads((run/'metrics.json').read_text())
        ts={d:dict(threshold=m['threshold'],source=m['threshold_source']) for d,m in metrics['devices'].items()}
        diagnostic=evaluate(name,2,ts)
        row[mode]={'forward':forward['aggregate'],'diagnostic':diagnostic['aggregate']}
    row['forward_delta']=row['event']['forward']['score']-row['none']['forward']['score']
    row['diagnostic_delta']=row['event']['diagnostic']['score']-row['none']['diagnostic']['score']
    results.append(row)
write_json(ROOT/'confirmation_results.json',results)
print(json.dumps(results,indent=2))
