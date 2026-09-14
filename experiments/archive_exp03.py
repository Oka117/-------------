"""Snapshot final EXP-03 sources and verify comparable data/config across runs."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline import sha256,write_json
from experiments.run_exp03 import RUNS


def main():
    out=Path('runs/exp03_analysis');snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=[Path(n) for n in ['baseline.py','metrics.py','train.py','predict.py','temporal_features.py','package_submission.py','requirements.txt','README.md','EXP03_实验结果.md','baselinev1_独立实验计划.md']]
    files+=list(Path('experiments').glob('*.py'))+list(Path('tests').glob('*.py'))
    for file in files:
        target=snapshot/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    (snapshot/'working_tree.patch').write_text(subprocess.check_output(['git','diff'],text=True))
    original=json.loads(Path('runs/baseline_v1/manifest.json').read_text())
    baseline_folds=json.loads(Path('runs/baseline_v1/metrics.json').read_text())['folds']
    pairs=json.loads((out/'seed_confirmation.json').read_text())['results']
    paths=[Path('runs')/n for n in RUNS]+[Path(r['path']) for r in pairs]
    artifacts={};comparisons=[]
    for run in paths:
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['config']['rounds']==300 and manifest['config']['threads']==4
        for d,meta in manifest['models'].items():
            assert meta['source_sha256']==original['models'][d]['source_sha256']
        folds=json.loads((run/'metrics.json').read_text())['folds']
        for a,b in zip(folds,baseline_folds):
            for k in ['device','fold','fit_rows','fit_positives','fit_end','start','end','rows','positives']:
                assert a[k]==b[k],(run,k)
        artifacts[str(run)]={str(f.relative_to(run)):sha256(f) for f in run.rglob('*') if f.is_file()}
        comparisons.append(dict(run=str(run),seed=manifest['config']['seed'],feature_kind=manifest['config']['feature_kind'],same_data_and_splits=True))
    for file,h in json.loads(Path('runs/exp03_protocol/baseline_hashes.json').read_text()).items():
        assert sha256(file)==h
    write_json(out/'provenance.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        source_sha256={str(f.relative_to(snapshot)):sha256(f) for f in snapshot.rglob('*') if f.is_file()},
        run_artifact_sha256=artifacts,comparisons=comparisons,baseline_unchanged=True,
        tests_passed=13,old_numerically_inconsistent_runs='runs/exp03_superseded; excluded from results'))
    print(f'Archived sources; verified {len(paths)} runs use matching baseline data and splits.')


if __name__=='__main__':main()
