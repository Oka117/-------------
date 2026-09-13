import tempfile
import unittest
from pathlib import Path
import numpy as np
from threshold_transfer import normal_percentile, shrink_threshold, decision_scores
from experiments.exp02 import diagnostics


class TransferTests(unittest.TestCase):
    def test_ecdf_ties_and_outside_range(self):
        ref=np.array([.1,.2,.2,.8])
        np.testing.assert_array_equal(normal_percentile([0,.1,.2,.5,.8,1],ref),[0,.25,.75,.75,1,1])
        with self.assertRaises(ValueError):
            normal_percentile([.5],[.2,.1])

    def test_shrink_endpoints_and_logit(self):
        sentinel=np.nextafter(1.,2.)
        self.assertEqual(shrink_threshold(sentinel,.4,0),sentinel)
        self.assertEqual(shrink_threshold(.2,sentinel,1),sentinel)
        self.assertAlmostEqual(shrink_threshold(.2,.8,.5),.5)
        self.assertTrue(np.isfinite(shrink_threshold(0,1,.5)))

    def test_saved_mapping_and_chunk_parity(self):
        ref=np.array([.1,.1,.5,.9])
        p=np.array([0,.1,.2,.5,.7,1.])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.save(root/'normal.npy',ref)
            meta=dict(score_transform='normal_percentile',normal_reference='normal.npy')
            full=decision_scores(p,meta,root)
            chunks=np.r_[decision_scores(p[:2],meta,root),decision_scores(p[2:],meta,root)]
            np.testing.assert_array_equal(full,chunks)
            np.testing.assert_array_equal(decision_scores(p,{},root),p)
            altered=p.copy(); altered[-1]=0
            np.testing.assert_array_equal(decision_scores(altered,meta,root)[:-1],full[:-1])

    def test_event_miss_and_short_window(self):
        b=dict(y=np.array([0,1,1,0]), timestamp=np.array(['a','b','c','d']))
        s,events,_=diagnostics(b,np.array([0,0,0,1]),'test',2,.5,'test')
        self.assertIsNone(events[0]['first_hit_delay_hours'])
        self.assertTrue(events[0]['missed_event'])
        self.assertEqual(s['fp'],1)
        _,events,_=diagnostics(b,np.array([0,0,1,0]),'test',2,.5,'test')
        self.assertEqual(events[0]['first_3h_recall'],.5)
        self.assertEqual(events[0]['first_hit_delay_hours'],1/3)


if __name__=='__main__':
    unittest.main()
