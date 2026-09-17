"""Causal probability postprocessing, with explicit per-model stream state."""
import numpy as np


def postprocess_probabilities(probabilities, threshold, config=None, state=None):
    """Return (binary alarms, new state); callers reset state at model/gap boundaries.

    Cold start: alarm off and no history. Smoothing uses available points (min=1).
    Equality triggers/maintains an alarm. No future inputs or labels are accepted.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite one-dimensional probabilities in [0,1]')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid threshold')
    config = config or {'mode': 'none'}
    state = state or {}
    mode = config.get('mode', 'none')
    pred = np.empty(len(p), dtype=np.int8)
    if mode == 'none':
        return (p >= threshold).astype(np.int8), {}
    if mode == 'hysteresis':
        low = float(config['low_threshold'])
        if not np.isfinite(low) or not 0 <= low <= threshold:
            raise ValueError('Require 0 <= low threshold <= high threshold')
        active = bool(state.get('active', False))
        for i, value in enumerate(p):
            active = bool(value >= (low if active else threshold))
            pred[i] = active
        return pred, {'active': active}
    if mode == 'smoothing':
        values, new_state = smooth_probabilities(p, config['window'], state)
        return (values >= threshold).astype(np.int8), new_state
    raise ValueError(f'Unknown postprocessor {mode}')


def smooth_probabilities(probabilities, window, state=None):
    if window not in (2, 3):
        raise ValueError('Smoothing window must be 2 or 3')
    history = list((state or {}).get('history', []))
    if len(history) > window - 1 or not np.isfinite(history).all() or any(x < 0 or x > 1 for x in history):
        raise ValueError('Invalid smoothing history')
    result = []
    for value in probabilities:
        history.append(float(value))
        result.append(sum(history) / len(history))
        history = history[-(window - 1):]
    return np.asarray(result), {'history': history}
