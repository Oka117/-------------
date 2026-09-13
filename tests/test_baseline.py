import unittest
import numpy as np
import pandas as pd
from metrics import event_weights, score, aggregate, choose_threshold
from baseline import event_safe_boundary, temporal_folds


class MetricsTests(unittest.TestCase):
    def test_official_example_and_singleton(self):
        np.testing.assert_array_equal(event_weights([0,1,1,1,1,0,1]), [0,7,5,3,1,0,7])

    def test_perfect_and_wrong(self):
        y = np.array([0,1,1,0])
        self.assertEqual(score(y,y)['score'], 100)
        self.assertEqual(score(y,1-y)['score'], 0)
        self.assertEqual(score(y,[0,0,0,0])['score'], 25)
        self.assertEqual(score(y,[1,1,1,1])['score'], 75)

    def test_no_positive_and_device_boundary(self):
        self.assertIsNone(score([0,0],[0,0])['score'])
        parts = [score([0,1],[0,1]), score([1,0],[0,0])]
        self.assertEqual(aggregate(parts)['score'], 62.5)

    def test_threshold_sweep_matches_brute_force(self):
        blocks = [dict(y=np.array([0,1,1,0,1]),p=np.array([.2,.6,.6,.8,.1])),
                  dict(y=np.array([1,0]),p=np.array([.7,.2]))]
        t = choose_threshold(blocks)
        def value(t):
            return aggregate([score(b['y'],b['p']>=t) for b in blocks])['score']
        candidates = [np.nextafter(1.,2.),0.,.1,.2,.6,.7,.8,1.]
        self.assertAlmostEqual(value(t),max(map(value,candidates)))

    def test_boundary_never_splits_event(self):
        self.assertEqual(event_safe_boundary([0,1,1,1,0],3),1)
        self.assertEqual(event_safe_boundary([0,1,1,1,0],4),4)

    def test_global_threshold_optimizes_aggregate_not_device_score(self):
        local = dict(y=np.array([0, 0, 0, 1]), p=np.array([.9, .8, .7, .1]))
        other_y = np.ones(100, dtype=np.int8)
        other = score(other_y, other_y)
        rows = len(local['y']) + len(other_y)
        weight = event_weights(local['y']).sum() + event_weights(other_y).sum()
        old = choose_threshold([local])
        new = choose_threshold([local], global_rows=rows, global_weight=weight)
        def value(t):
            return aggregate([score(local['y'], local['p'] >= t), other])['score']
        candidates = [np.nextafter(1., 2.), .9, .8, .7, .1]
        self.assertEqual(old, .1)
        self.assertEqual(new, candidates[0])
        self.assertGreater(value(new), value(old))
        self.assertAlmostEqual(value(new), max(map(value, candidates)))

    def test_global_threshold_randomized_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            blocks = [dict(y=np.r_[0, 1, rng.integers(0, 2, size=8)],
                           p=rng.integers(0, 6, size=10) / 5) for _ in range(2)]
            other_y = np.r_[0, 1, rng.integers(0, 2, size=28)]
            other = score(other_y, rng.integers(0, 2, size=30))
            rows = sum(len(b['y']) for b in blocks) + len(other_y)
            weight = sum(event_weights(b['y']).sum() for b in blocks) + event_weights(other_y).sum()
            t = choose_threshold(blocks, global_rows=rows, global_weight=weight)
            def value(t):
                return aggregate([score(b['y'], b['p'] >= t) for b in blocks] + [other])['score']
            candidates = np.r_[np.nextafter(1., 2.), np.unique(np.concatenate([b['p'] for b in blocks]))]
            self.assertAlmostEqual(value(t), max(map(value, candidates)))

    def test_global_denominators_validation_and_legacy_equivalence(self):
        b = dict(y=np.array([0, 1]), p=np.array([.6, .2]))
        self.assertEqual(choose_threshold([b]),
                         choose_threshold([b], global_rows=2, global_weight=7))
        for kwargs in [dict(global_rows=2), dict(global_weight=7),
                       dict(global_rows=0, global_weight=7),
                       dict(global_rows=2, global_weight=np.nan),
                       dict(global_rows=np.inf, global_weight=7),
                       dict(global_rows=1, global_weight=7),
                       dict(global_rows=2, global_weight=6)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                choose_threshold([b], **kwargs)
        self.assertEqual(choose_threshold([], global_rows=2, global_weight=7), .5)
        normal = dict(y=np.array([0, 0]), p=np.array([.6, .2]))
        self.assertEqual(choose_threshold([normal], global_rows=4, global_weight=7), .5)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            score([0,2],[0,1])

    def test_temporal_folds_keep_events_whole_and_gap(self):
        df = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=720, freq='20min')})
        y = np.zeros(720, dtype=np.int8)
        y[210:230] = 1  # Crosses January 4 (index 216).
        folds = list(temporal_folds(df, y, ['2024-01-04','2024-01-06','2024-01-08'],72))
        self.assertEqual(folds[0][2],210)
        for _, fit_end, start, end in folds:
            self.assertGreaterEqual(start-fit_end,72)
            self.assertLess(fit_end,end)
            self.assertFalse(y[start-1] == y[start] == 1)
        self.assertEqual(folds[-1][-1],len(y))

    def test_threshold_ties_and_degenerate_calibration(self):
        self.assertEqual(choose_threshold([]),0.5)
        self.assertEqual(choose_threshold([dict(y=np.array([0,0]),p=np.array([.2,.3]))]),0.5)
        b = dict(y=np.array([0,1]),p=np.array([.5,.5]))
        self.assertEqual(choose_threshold([b]),.5)


if __name__ == '__main__':
    unittest.main()
