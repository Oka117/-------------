"""Package baseline_v1 code and six trained models without raw data."""
import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    run = root / 'runs/baseline_v1'
    manifest = json.loads((run / 'manifest.json').read_text())
    expected = {'P106A', 'P202A', 'P310A', 'P310B', 'P412B', 'P601B'}
    if set(manifest['devices']) != expected:
        raise ValueError('The submission must contain all six devices')
    files = [root / name for name in ['predict.py', 'train.py', 'baseline.py', 'metrics.py',
                                     'temporal_features.py', 'package_submission.py', 'README.md', 'requirements.txt']]
    files += [run / 'manifest.json', run / 'config.json']
    for device in sorted(expected):
        model = manifest['models'][device]['path']
        if model:
            files.append(run / model)
        if manifest['models'][device].get('history_path'):
            files.append(run / manifest['models'][device]['history_path'])
    if not all(p.is_file() for p in files):
        raise FileNotFoundError('A required source or model file is missing')
    output = run / 'submission/submission_code.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to silently replace an earlier submission package.
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, arcname=str(path.relative_to(root)))
    print(output)


if __name__ == '__main__':
    main()
