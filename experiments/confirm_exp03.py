"""Two additional paired seeds for the family chosen exclusively on forward fold1."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline import write_json
from experiments.run_exp03 import RUNS
from experiments.analyze_exp03 import read_blocks,calibrate,evaluate


def main():
    out=Path('runs/exp03_analysis')
    selection=json.loads((out/'selection.json').read_text())
    chosen=selection['chosen']
    if chosen=='baseline_v1':
        write_json(out/'seed_confirmation.json',dict(selected=chosen,reason='No forward improvement; no candidate to confirm',results=[]))
        return
    records=[]
    for seed in selection['second_stage_seeds']:
        jobs=[]
        for label,kind in [('raw','raw'),(chosen,RUNS[chosen])]:
            path=Path('runs/exp03_confirmation')/f'{label}_seed{seed}'
            log=Path('runs/exp03_logs')/f'confirm_{label}_seed{seed}.log'
            command=['.venv/bin/python','train.py','--feature-kind',kind,'--seed',str(seed),'--output',str(path),'--threads','4','--rounds','300']
            handle=log.open('x')
            proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
            jobs.append((label,path,proc,handle))
            print('RUN',' '.join(command),flush=True)
        for label,path,proc,handle in jobs:
            status=proc.wait();handle.close()
            if status:raise RuntimeError(f'Failed {path}')
            metadata,history=read_blocks(path,[0,1])
            forward=evaluate({d:bs[1:] for d,bs in history.items()},calibrate({d:bs[:1] for d,bs in history.items()}))
            _,holdout=read_blocks(path,[2])
            diagnosis=evaluate(holdout,calibrate(history))
            records.append(dict(seed=seed,family=label,path=str(path),forward=forward,fold2=diagnosis,
                                seconds=metadata['elapsed_seconds'],peak_rss_mib=metadata['peak_rss_mib']))
            print('DONE',label,seed,'forward',forward['aggregate']['score'],'fold2',diagnosis['aggregate']['score'],flush=True)
        write_json(out/'seed_confirmation.json',dict(selected=chosen,parallel_pair_per_seed=True,results=records))


if __name__=='__main__':main()
