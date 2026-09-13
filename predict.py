"""Generate predictions preserving test timestamps and row order."""
import argparse
import json
import zipfile
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from baseline import read_features, probability
from threshold_transfer import decision_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', '--data_dir', default='data')
    ap.add_argument('--model-dir', default=str(Path(__file__).resolve().parent / 'runs/baseline_v1'))
    ap.add_argument('--output', default='runs/baseline_v1/predictions')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--pump-column', default='pump_id', help='Pump column name; pump_id is the provisional user-selected default.')
    ap.add_argument('--zip', action='store_true', help='Pack prediction CSV files into result.zip')
    args = ap.parse_args()
    if args.threads < 1 or not args.pump_column or args.pump_column in ['timestamp', 'label']:
        ap.error('Use positive threads and a nonempty, distinct pump column')
    manifest = json.loads((Path(args.model_dir)/'manifest.json').read_text())
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory is not empty. Choose a new directory.')
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for device in manifest['devices']:
        meta = manifest['models'][device]
        df, cols = read_features(Path(args.data_dir)/device/f'{device}_test.csv')
        if cols != meta['features']:
            raise ValueError(f'{device}: test feature schema differs from training')
        if pd.Timestamp(df.timestamp.iloc[0]) <= pd.Timestamp(meta['train_end']):
            raise ValueError(f'{device}: test overlaps training history')
        model = lgb.Booster(model_file=str(Path(args.model_dir)/meta['path'])) if meta['path'] else None
        p = probability(model, meta['constant'], df[cols], args.threads)
        if not np.isfinite(p).all():
            raise ValueError(f'{device}: invalid model predictions')
        decision = decision_scores(p, meta, Path(args.model_dir))
        result = pd.DataFrame(dict(timestamp=df.timestamp, label=(decision >= meta['threshold']).astype(np.int8)))
        if args.pump_column:
            result.insert(0, args.pump_column, device)
        path = out/f'{device}_predict.csv'
        result.to_csv(path, index=False)
        pd.DataFrame(dict(timestamp=df.timestamp, probability=p)).to_csv(out/f'{device}_probabilities.csv', index=False)
        files.append(path)
        print(f'{device}: {len(result)} rows, {int(result.label.sum())} predicted abnormal', flush=True)
    if args.zip:
        with zipfile.ZipFile(out/'result.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for path in files:
                z.write(path, arcname=path.name)
    print('Prediction files:', out.resolve())


if __name__ == '__main__':
    main()
