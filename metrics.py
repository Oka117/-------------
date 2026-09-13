"""Competition score. Compute weights separately for each device/time block."""
import numpy as np


def binary(values):
    a = np.asarray(values)
    if a.ndim != 1 or not np.isin(a, [0, 1]).all():
        raise ValueError('Expected a one-dimensional binary array')
    return a.astype(np.int8)


def event_weights(labels):
    y = binary(labels)
    w = np.zeros(len(y), dtype=float)
    edges = np.diff(np.r_[0, y, 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        w[start:end] = np.linspace(7, 1, end - start) if end - start > 1 else 7
    return w


def score(labels, predictions, weights=None):
    y, p = binary(labels), binary(predictions)
    if len(y) == 0 or y.shape != p.shape:
        raise ValueError('Nonempty labels/predictions must have the same shape')
    w = event_weights(y) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any() or (w[y == 0] != 0).any() or (w[y == 1] <= 0).any():
        raise ValueError('Invalid weights')
    correct = int((y == p).sum())
    hit, total = float(w[p == 1].sum()), float(w.sum())
    neg = int((y == 0).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    return dict(rows=len(y), positives=int(y.sum()), correct=correct,
                accuracy=correct / len(y), weighted_hit=hit, weighted_total=total,
                weighted_recall=hit / total if total else None,
                score=50 * correct / len(y) + 50 * hit / total if total else None,
                false_positive_rate=fp / neg if neg else None,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / int(y.sum()) if y.sum() else None)


def aggregate(parts):
    n = sum(s['rows'] for s in parts)
    correct = sum(s['correct'] for s in parts)
    hit = sum(s['weighted_hit'] for s in parts)
    total = sum(s['weighted_total'] for s in parts)
    return dict(rows=n, accuracy=correct / n if n else None,
                weighted_recall=hit / total if total else None,
                score=50 * correct / n + 50 * hit / total if n and total else None)


def choose_threshold(blocks, *, global_rows=None, global_weight=None):
    """Exact sweep; optionally optimize a contribution to a shared total score.

    Supply both denominators from the same eligible calibration pool. Omitting
    both preserves the original block-local objective and fallback behavior.
    Ties prefer fewer alarms.
    """
    shared = global_rows is not None or global_weight is not None
    if shared:
        if (global_rows is None or global_weight is None
                or not np.isfinite(global_rows) or not np.isfinite(global_weight)
                or global_rows <= 0 or global_weight <= 0):
            raise ValueError('Supply both finite, positive global denominators')
    if not blocks:
        return 0.5
    y = np.concatenate([b['y'] for b in blocks])
    p = np.concatenate([b['p'] for b in blocks])
    w = np.concatenate([event_weights(b['y']) for b in blocks])
    if shared and (global_rows < len(y)
                   or global_weight < w.sum() - 1e-10 * max(1., w.sum())):
        raise ValueError('Global denominators must include the local blocks')
    if not len(y) or not w.sum() or not (y == 0).any():
        return 0.5
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and in [0,1]')
    order = np.argsort(-p, kind='stable')
    ps, ys, ws = p[order], y[order], w[order]
    # Starting with all-negative predictions, flip tied probabilities together.
    rows = global_rows if shared else len(y)
    weight = global_weight if shared else w.sum()
    delta = 50 * (2 * ys.astype(float) - 1) / rows + 50 * ws / weight
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    gains = np.r_[0., np.cumsum(delta)[ends]]
    thresholds = np.r_[np.nextafter(1., 2.), ps[ends]]
    return float(thresholds[int(np.argmax(gains))])
