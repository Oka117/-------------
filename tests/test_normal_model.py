import tempfile
import unittest
from pathlib import Path
import numpy as np
from normal_model import fit_normal_model, normal_error, bounded_error, fit_cdf, apply_cdf, fusion_score
from experiments.exp06 import threshold_for, diagnostics


class NormalModelTests(unittest.TestCase):
    def test_abnormal_rows_do_not_enter_fit(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(120,6)); y=np.r_[np.zeros(100),np.ones(20)]
        a=fit_normal_model(x,y)
        x[100:]=1e8
        b=fit_normal_model(x,y)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])

    def test_residual_detects_orthogonal_novelty_and_roundtrip(self):
        t=np.linspace(-1,1,100)
        x=np.column_stack([t,t,np.ones(100)])
        state=fit_normal_model(x,np.zeros(100))
        self.assertEqual(state['components'].shape[1],1)
        self.assertLess(normal_error(state,[[0,0,1]])[0],1e-20)
        self.assertGreater(normal_error(state,[[2,-2,1]])[0],1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz'
            np.savez(path,**state)
            with np.load(path,allow_pickle=False) as loaded:
                np.testing.assert_allclose(normal_error(loaded,x),normal_error(state,x),atol=1e-25)
        # Current-time-only scoring: chunks and future perturbations preserve earlier scores.
        full=normal_error(state,x)
        chunks=np.concatenate([normal_error(state,x[:37]),normal_error(state,x[37:])])
        np.testing.assert_allclose(full,chunks,atol=1e-25)
        x[80:]+=100
        np.testing.assert_array_equal(full[:80],normal_error(state,x)[:80])

    def test_tail_ties_and_missing_supervision(self):
        ref=fit_cdf([1,1,2,3])
        np.testing.assert_array_equal(apply_cdf(ref,[0,1,2,4]),[0,.25,.625,1])
        threshold,source=threshold_for(np.zeros(10),np.ones(10))
        self.assertGreater(threshold,1)
        self.assertIn('no_recall_evidence',source)
        maps=dict(supervised_available=np.array(False))
        error=np.array([0.,1.,100.])
        np.testing.assert_array_equal(fusion_score(error,np.ones(3),maps,.5),bounded_error(error))

    def test_missed_event_and_short_window(self):
        result,_=diagnostics(np.array([0,1,1,0]),np.zeros(4),.5,np.arange(4))
        self.assertTrue(result['events'][0]['missed_event'])
        self.assertIsNone(result['events'][0]['delay_hours'])
        self.assertEqual(result['events'][0]['recall_first_6h'],0)


if __name__=='__main__':
    unittest.main()
