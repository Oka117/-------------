"""Train per-device baselines with calibration folds and a held-out final block."""
import argparse
import platform
import resource
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from baseline import read_train, temporal_folds, fit_model, probability, write_json, sha256
from metrics import aggregate, choose_threshold, event_weights, score
from temporal_features import WINDOWS, fit_recipe, build_features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', '--data_dir', default='data')
    ap.add_argument('--output', default='runs/baseline_v1')
    ap.add_argument('--devices', nargs='+', help='Default: all devices found under data-dir')
    ap.add_argument('--rounds', type=int, default=300)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--feature-kind', choices=['raw', *WINDOWS], default='raw')
    ap.add_argument('--gap', type=int, default=72, help='Excluded training points before each validation block')
    ap.add_argument('--fold-starts', nargs=3, default=['2024-04-01', '2024-07-01', '2024-09-01'],
                    metavar='DATE', help='First two blocks calibrate thresholds; third is holdout')
    args = ap.parse_args()
    if args.rounds < 1 or args.threads < 1 or args.gap < 0:
        ap.error('rounds/threads must be positive; gap must be nonnegative')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory is not empty. Choose a new run directory.')
    devices = args.devices or sorted(p.name for p in Path(args.data_dir).iterdir() if p.is_dir() and (p / f'{p.name}_train.csv').exists())
    if not devices or len(devices) != len(set(devices)):
        ap.error('Expected one or more unique devices')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir()
    (out / 'validation').mkdir()
    started = time.perf_counter()
    manifest = dict(config=vars(args), devices=devices, versions=dict(python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, lightgbm=lgb.__version__), models={})
    write_json(out / 'config.json', manifest)
    all_blocks, fold_report = {}, []
    (out / 'feature_recipes').mkdir()
    (out / 'fold_models').mkdir()
    for device in devices:
        df, cols, y = read_train(args.data_dir, device)
        x = df[cols]
        blocks = []
        for i, fit_end, start, end in temporal_folds(df, y, args.fold_starts, args.gap):
            t0 = time.perf_counter()
            recipe = None
            if args.feature_kind != 'raw':
                recipe = fit_recipe(df[cols].iloc[:fit_end], y[:fit_end], args.feature_kind, args.rounds, args.threads, args.seed)
                x = build_features(df[cols].iloc[:end], recipe)
                write_json(out/'feature_recipes'/f'{device}_fold{i}.json', recipe)
            model, constant = fit_model(x.iloc[:fit_end], y[:fit_end], args.rounds, args.threads, args.seed)
            if args.feature_kind != 'raw' and model is not None:
                model.save_model(str(out/'fold_models'/f'{device}_fold{i}.txt'))
            p0 = time.perf_counter()
            p = probability(model, constant, x.iloc[start:end], args.threads)
            inference_seconds = time.perf_counter()-p0
            b = dict(y=y[start:end], p=p, eligible=model is not None, fold=i,
                     timestamp=df.timestamp.iloc[start:end].to_numpy())
            blocks.append(b)
            row = dict(device=device, fold=i, role='holdout' if i == 2 else 'calibration',
                       fit_rows=fit_end, fit_positives=int(y[:fit_end].sum()),
                       fit_end=df.timestamp.iloc[fit_end - 1], start=df.timestamp.iloc[start],
                       end=df.timestamp.iloc[end - 1], rows=end-start, positives=int(b['y'].sum()),
                       model='lightgbm' if model else f'constant_{constant:g}', seconds=time.perf_counter()-t0,
                       inference_seconds=inference_seconds, feature_count=x.shape[1],
                       selection_source=recipe['selection_source'] if recipe else None)
            fold_report.append(row)
            print(f"{device} fold={i} {row['role']} {row['model']} train_pos={row['fit_positives']} val_pos={row['positives']} {row['seconds']:.1f}s", flush=True)
        all_blocks[device] = blocks
        del df, x, model
    # Holdout labels/probabilities NEVER enter threshold selection or round selection.
    pooled = [b for blocks in all_blocks.values() for b in blocks[:2] if b['eligible']]
    pooled_threshold = choose_threshold(pooled)
    report = dict(folds=fold_report, pooled_threshold=pooled_threshold, devices={},
                  notes=['Formula reproduced from competition page; no official scoring script supplied.',
                         'Holdout starts at the third fold boundary; no early stopping or threshold tuning on holdout.',
                         'No-positive block score is null; aggregate sums device-local weights.',
                         'Fallback thresholds transfer across devices without a calibration guarantee.'])
    holdout_scores = []
    for device, blocks in all_blocks.items():
        calibration = [b for b in blocks[:2] if b['eligible']]
        events = sum(int((np.diff(np.r_[0, b['y'], 0]) == 1).sum()) for b in calibration)
        enough = events >= 2 and any((b['y'] == 0).any() for b in calibration)
        threshold = choose_threshold(calibration) if enough else pooled_threshold
        source = 'device_calibration' if enough else 'pooled_calibration' if any(b['y'].sum() for b in pooled) else 'default_0.5_no_calibration_positives'
        scores = []
        for b in blocks:
            pred = (b['p'] >= threshold).astype(np.int8)
            scores.append(score(b['y'], pred))
            pd.DataFrame(dict(timestamp=b['timestamp'], label=b['y'], probability=b['p'], prediction=pred,
                              event_weight=event_weights(b['y']))).to_csv(out/'validation'/f'{device}_fold{b["fold"]}.csv', index=False)
        holdout_scores.append(scores[-1])
        report['devices'][device] = dict(threshold=threshold, threshold_source=source,
                                        calibration_events=events, blocks=scores,
                                        holdout_all_zero=score(blocks[-1]['y'], np.zeros(len(blocks[-1]['y']))),
                                        holdout_all_one=score(blocks[-1]['y'], np.ones(len(blocks[-1]['y']))))
        # Refit on all labeled history only after threshold and holdout evaluation.
        t0 = time.perf_counter()
        df, cols, y = read_train(args.data_dir, device)
        recipe = None
        x = df[cols]
        if args.feature_kind != 'raw':
            recipe = fit_recipe(x, y, args.feature_kind, args.rounds, args.threads, args.seed)
            x = build_features(x, recipe)
            write_json(out/'feature_recipes'/f'{device}_final.json', recipe)
            df.tail(72).to_csv(out/'models'/f'{device}_history.csv', index=False)
        model, constant = fit_model(x, y, args.rounds, args.threads, args.seed)
        model_path = f'models/{device}.txt'
        if model is not None:
            model.save_model(str(out/model_path))
            pd.DataFrame(dict(feature=list(x.columns), gain=model.feature_importance('gain'))).sort_values('gain', ascending=False).to_csv(out/'models'/f'{device}_importance.csv', index=False)
        manifest['models'][device] = dict(features=list(x.columns), raw_features=cols, feature_recipe=recipe,
            history_path=f'models/{device}_history.csv' if recipe else None, threshold=threshold, threshold_source=source,
            path=model_path if model is not None else None, constant=constant,
            train_end=df.timestamp.iloc[-1], train_rows=len(df),
            source_sha256={suffix:sha256(Path(args.data_dir)/device/f'{device}_{suffix}.csv') for suffix in ['train', 'train_label']})
        report['devices'][device]['final_fit_seconds'] = time.perf_counter()-t0
        print(f"{device} threshold={threshold:.6g} ({source}) holdout={scores[-1]['score']} final fit={time.perf_counter()-t0:.1f}s", flush=True)
    report['holdout_aggregate'] = aggregate(holdout_scores)
    report['feature_kind'] = args.feature_kind
    report['elapsed_seconds'] = time.perf_counter()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_mib'] = rss / (1024**2 if platform.system() == 'Darwin' else 1024)
    write_json(out/'manifest.json', manifest)
    write_json(out/'metrics.json', report)
    print(f"Done: {out.resolve()} holdout={report['holdout_aggregate']} elapsed={report['elapsed_seconds']:.1f}s peak_rss={report['peak_rss_mib']:.0f}MiB", flush=True)


if __name__ == '__main__':
    main()
