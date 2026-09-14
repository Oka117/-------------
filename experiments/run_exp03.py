"""Run predeclared independent EXP-03 feature families sequentially on CPU."""
import argparse
import subprocess
import time
from pathlib import Path

RUNS={'exp03a_history_deviation':'mean_deviation','exp03b_lag_difference':'lag_difference',
      'exp03c_history_std':'history_std','exp03d_standardized_deviation':'standardized_deviation'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',nargs='+',choices=list(RUNS),default=list(RUNS))
    args=ap.parse_args()
    logs=Path('runs/exp03_logs');logs.mkdir(exist_ok=True)
    for name in args.runs:
        kind=RUNS[name]
        cmd=['.venv/bin/python','train.py','--feature-kind',kind,'--output',f'runs/{name}','--threads','4','--rounds','300','--seed','42']
        print('RUN',' '.join(cmd),flush=True)
        with (logs/f'{name}.log').open('x') as log:
            process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            if process.wait():raise RuntimeError(f'Failed {name}; inspect log')


if __name__=='__main__':main()
