"""EXP-02 score mappings; shared by offline evaluation and inference."""
import numpy as np


def normal_percentile(probabilities, sorted_normal):
    reference = np.asarray(sorted_normal, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
        raise ValueError('Nonempty finite normal reference required')
    if (np.diff(reference) < 0).any() or (reference < 0).any() or (reference > 1).any():
        raise ValueError('Reference must be sorted probabilities')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probabilities')
    # Right-continuous empirical CDF: ties flip together, no interpolation.
    return np.searchsorted(reference, p, side='right') / len(reference)


def shrink_threshold(device_threshold, shared_threshold, alpha):
    """alpha=0 keeps device threshold; alpha=1 fully transfers shared threshold."""
    if alpha not in (0., .5, 1.):
        raise ValueError('Expected alpha in {0, 0.5, 1}')
    if alpha == 0:
        return float(device_threshold)  # Preserve all-negative >1 sentinel.
    if alpha == 1:
        return float(shared_threshold)
    a, b = np.clip([device_threshold, shared_threshold], 1e-12, 1-1e-12)
    logit = (1-alpha)*np.log(a/(1-a)) + alpha*np.log(b/(1-b))
    return float(1/(1+np.exp(-logit)))


def decision_scores(probabilities, meta, model_dir):
    if meta.get('score_transform') == 'normal_percentile':
        return normal_percentile(probabilities, np.load(model_dir / meta['normal_reference']))
    return np.asarray(probabilities)
