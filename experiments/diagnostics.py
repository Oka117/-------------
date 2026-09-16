"""Reusable diagnostics; event and false-alarm duration count sampled 20-minute bins."""
import numpy as np
from metrics import event_weights, score


def segments(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def diagnose(frame, threshold):
    y = frame.label.to_numpy()
    pred = (frame.probability.to_numpy() >= threshold).astype(np.int8)
    weights = frame.event_weight.to_numpy() if 'event_weight' in frame else event_weights(y)
    result = score(y, pred, weights)
    result.update(FP=int(((y == 0) & (pred == 1)).sum()), FN=int(((y == 1) & (pred == 0)).sum()))
    events = []
    for a, b in segments(y):
        hits = np.flatnonzero(pred[a:b])
        events.append(dict(start=str(frame.timestamp.iloc[a]), end=str(frame.timestamp.iloc[b-1]), rows=int(b-a),
            missed_event=not bool(len(hits)), delay_minutes=int(hits[0])*20 if len(hits) else None,
            recall_3h=float(pred[a:min(b,a+9)].mean()), recall_6h=float(pred[a:min(b,a+18)].mean()),
            weighted_recall=float(np.dot(weights[a:b],pred[a:b])/weights[a:b].sum())))
    alarms = [dict(start=str(frame.timestamp.iloc[a]), end=str(frame.timestamp.iloc[b-1]),
        rows=int(b-a), minutes=int(b-a)*20) for a,b in segments((y == 0) & (pred == 1))]
    result.update(events=events, false_alarm_segments=alarms,
        long_false_positive_points=sum(a['rows'] for a in alarms if a['rows'] >= 18))
    return result, pred
