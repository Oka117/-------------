import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from baseline import fit_model
from experiments import exp05


class CapacityTests(unittest.TestCase):
    def test_defaults_and_single_parameter_changes_reach_lightgbm(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame(rng.normal(size=(500, 4)), columns=list('abcd'))
        y = (x.a + x.b > 0).to_numpy(dtype=np.int8)
        baseline, _ = fit_model(x, y, 5, 1, 42)
        explicit, _ = fit_model(x, y, 5, 1, 42, 100, 15)
        np.testing.assert_array_equal(baseline.predict(x, num_threads=1), explicit.predict(x, num_threads=1))
        for minimum, leaves in [(20, 15), (100, 7)]:
            model, _ = fit_model(x, y, 5, 1, 42, minimum, leaves)
            self.assertEqual(model.params['min_data_in_leaf'], minimum)
            self.assertEqual(model.params['num_leaves'], leaves)
            self.assertEqual(model.current_iteration(), 5)
            self.assertTrue(all(tree['num_leaves'] <= leaves for tree in model.dump_model()['tree_info']))

    def test_single_class_still_returns_constant(self):
        model, constant = fit_model(pd.DataFrame({'x': [1, 2]}), np.array([0, 0]), 100, 1, 42, 20, 31)
        self.assertIsNone(model)
        self.assertEqual(constant, 0)

    def test_diagnostics_serialize_and_mark_missed_short_events(self):
        df = pd.DataFrame(dict(timestamp=pd.date_range('2024-07-01',periods=24,freq='20min').astype(str),
                               label=[0]*18+[1,1,0,0,1,1],probability=[1]*18+[0,0,0,0,1,0]))
        with TemporaryDirectory() as temp, patch.object(exp05,'OUT',Path(temp)), \
             patch.object(exp05,'thresholds',return_value={'P':(.5,'pooled_calibration',1)}), \
             patch.object(exp05,'load',return_value={'P':df}):
            result=exp05.evaluate(Path(temp),'test',1,[0])
            self.assertEqual(result['long_false_alarm_points'],18)
            self.assertEqual(result['missed_events'],1)
            events=pd.read_csv(Path(temp)/'test/forward_fold1_events.csv')
            self.assertTrue(pd.isna(events.iloc[0].delay_minutes))
            self.assertEqual(events.iloc[1].recall_3h,.5)
            self.assertEqual(events.iloc[1].recall_6h,.5)


if __name__ == '__main__':
    unittest.main()
