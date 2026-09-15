"""EXP06 normal-only robust PCA; portable NumPy state, no pickle."""
import numpy as np


def fit_normal_model(x, y, variance=0.95):
    x = np.asarray(x, dtype=np.float64)
    normal = x[np.asarray(y) == 0]
    if len(normal) < 2 or not 0 < variance < 1:
        raise ValueError('Need at least two normal rows and variance in (0,1)')
    center = np.median(normal, axis=0)
    q25, q75 = np.quantile(normal, [0.25, 0.75], axis=0)
    # For zero IQR, use normal-training standard deviation; constants use 1.
    scale = q75 - q25
    std = normal.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.))
    z = (normal - center) / scale
    mean = z.mean(axis=0)
    z -= mean
    values, vectors = np.linalg.eigh(z.T @ z / (len(z) - 1))
    values, vectors = np.maximum(values[::-1], 0), vectors[:, ::-1]
    k = int(np.searchsorted(np.cumsum(values), variance * values.sum()) + 1)
    k = min(k, x.shape[1])
    return dict(center=center, scale=scale, mean=mean, components=vectors[:, :k],
                normal_rows=np.array(len(normal)), retained_variance=np.array(
                    values[:k].sum() / values.sum() if values.sum() else 1.))


def normal_error(state, x):
    x = np.asarray(x, dtype=np.float64)
    result = []
    for start in range(0, len(x), 2048):
        z = (x[start:start+2048] - state['center']) / state['scale'] - state['mean']
        residual = z - (z @ state['components']) @ state['components'].T
        result.append(np.mean(residual * residual, axis=1))
    return np.concatenate(result) if result else np.empty(0)


def bounded_error(error):
    return -np.expm1(-np.log1p(error))


def fit_cdf(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Insufficient finite historical normal scores')
    return np.sort(values)


def apply_cdf(reference, values):
    # Equal observed values get their midrank; all larger tail scores saturate.
    return (np.searchsorted(reference, values, side='left') +
            np.searchsorted(reference, values, side='right')) / (2 * len(reference))


def fusion_score(error, supervised, maps, alpha):
    if not bool(maps['supervised_available']):
        return bounded_error(error)
    return ((1 - alpha) * apply_cdf(maps['normal_cdf'], error) +
            alpha * apply_cdf(maps['supervised_cdf'], supervised))
