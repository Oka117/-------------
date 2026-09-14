import unittest
import numpy as np
from baseline import training_weights, fit_model


class TrainingWeightTests(unittest.TestCase):
    def test_equal_event_mass_and_singletons(self):
        y = np.array([0, 1, 1, 1, 1, 0, 1, 0])
        for c in (2, 4):
            a = training_weights(y, 'class', c)
            b = training_weights(y, 'event', c)
            np.testing.assert_allclose(b, [1, c*7/4, c*5/4, c*3/4, c/4, 1, c, 1])
            self.assertAlmostEqual(a.sum(), b.sum())
            self.assertEqual(b[6], c)
        self.assertIsNone(training_weights(y))

    def test_prefix_does_not_use_future_labels(self):
        y = np.array([0, 1, 1, 0, 0, 1])
        a = training_weights(y[:4], 'event', 2)
        y[4:] = 1-y[4:]
        np.testing.assert_array_equal(a, training_weights(y[:4], 'event', 2))

    def test_reject_invalid_weights_even_for_constant_model(self):
        for w in ([1], [1, 0], [1, np.nan]):
            with self.assertRaises(ValueError):
                fit_model(np.zeros((2, 1)), np.zeros(2), 1, 1, 42, w)

    def test_weight_reaches_training(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(400, 2))
        y = np.tile([0, 0, 0, 1], 100)
        a, _ = fit_model(x, y, 1, 1, 42)
        b, _ = fit_model(x, y, 1, 1, 42, training_weights(y, 'class', 4))
        self.assertGreater(b.predict(x).mean(), a.predict(x).mean() + .2)
