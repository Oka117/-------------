"""Shared IO and LightGBM helpers. Raw current-time features only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import binary, event_weights


def read_features(path):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    features = [c for c in columns if c.startswith('feature_')]
    if set(columns) != {'timestamp', *features} or not features:
        raise ValueError(f'{path}: expected timestamp and feature_* columns')
    df = pd.read_csv(path, dtype={c: 'float32' for c in features})
    t = pd.to_datetime(df.timestamp, errors='raise')
    if not len(df) or t.isna().any() or t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError(f'{path}: invalid timestamps')
    if not t.diff().iloc[1:].eq(pd.Timedelta(minutes=20)).all():
        raise ValueError(f'{path}: expected continuous 20-minute intervals')
    if not np.isfinite(df[features].to_numpy()).all():
        raise ValueError(f'{path}: nonfinite features; explicit preprocessing required')
    return df, features


def read_train(data_dir, device):
    folder = Path(data_dir) / device
    df, features = read_features(folder / f'{device}_train.csv')
    labels = pd.read_csv(folder / f'{device}_train_label.csv')
    if not labels.timestamp.equals(df.timestamp):
        raise ValueError(f'{device}: labels do not align with feature timestamps')
    return df, features, binary(labels.label)


def event_safe_boundary(y, index):
    """Move a boundary to the beginning of an event it would split."""
    while 0 < index < len(y) and y[index - 1] == 1 and y[index] == 1:
        index -= 1
    return index


def temporal_folds(df, y, starts, gap):
    t = pd.DatetimeIndex(pd.to_datetime(df.timestamp))
    boundaries = [event_safe_boundary(y, int(t.searchsorted(pd.Timestamp(s)))) for s in starts] + [len(y)]
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('Fold dates must produce ordered, nonempty validation blocks')
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        fit_end = event_safe_boundary(y, max(0, start - gap))
        if fit_end == 0:
            raise ValueError('No training data before validation/gap')
        yield i, fit_end, start, end


def training_weights(y, mode='none', positive_weight=2.0):
    """Training-prefix labels only; singleton events have weight c in both modes."""
    y = binary(y)
    if mode not in ('none', 'class', 'event') or not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError('Invalid training weight configuration')
    if mode == 'none':
        return None
    w = np.ones(len(y), dtype=float)
    w[y == 1] = positive_weight
    if mode == 'event':
        ew = event_weights(y)
        edges = np.diff(np.r_[0, y, 0])
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start > 1:
                w[start:end] = positive_weight * ew[start:end] / 4.0
    return w


def fit_model(x, y, rounds, threads, seed, sample_weight=None):
    import lightgbm as lgb
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != np.asarray(y).shape or not np.isfinite(sample_weight).all() or (sample_weight <= 0).any():
            raise ValueError('Expected finite positive sample weights aligned with labels')
    if np.unique(y).size < 2:
        return None, float(y[0])
    params = dict(objective='binary', metric='binary_logloss', learning_rate=0.05,
                  num_leaves=15, min_data_in_leaf=100, lambda_l2=5.0,
                  feature_fraction=0.9, max_bin=127, verbosity=-1,
                  num_threads=threads, seed=seed, deterministic=True, force_col_wise=True)
    model = lgb.train(params, lgb.Dataset(x, label=y, weight=sample_weight), num_boost_round=rounds)
    return model, None


def probability(model, constant, x, threads):
    if model is None:
        return np.full(len(x), constant, dtype=float)
    return model.predict(x, num_threads=threads)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
