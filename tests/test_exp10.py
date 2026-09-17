import unittest
import numpy as np
from experiments.exp10_thresholds import choose_threshold
from experiments.exp10_postprocessing import smooth_probabilities
from baseline import event_safe_boundary
from metrics import event_weights

class Exp10Tests(unittest.TestCase):
    def test_exact_global_sweep_with_ties(self):
        y=np.array([0,1,1,0,1,0]); p=np.array([.2,.2,.8,.8,.1,.1])
        t=choose_threshold([dict(y=y,p=p)],global_rows=20,global_weight=30)
        w=event_weights(y)
        candidates=np.r_[np.nextafter(1.,2.),np.unique(p)[::-1]]
        gains=[50*((y==(p>=x)).sum())/20+50*w[p>=x].sum()/30 for x in candidates]
        self.assertEqual(t,candidates[np.argmax(gains)])
    def test_causal_chunk_and_cold_start(self):
        p=np.random.default_rng(42).random(412)
        full=smooth_probabilities(p,2)[0]
        expected=np.r_[p[0],(p[1:]+p[:-1])/2]
        np.testing.assert_array_equal(full,expected)
        state=None; parts=[]
        for i in range(0,len(p),137):
            part,state=smooth_probabilities(p[i:i+137],2,state); parts.extend(part)
        np.testing.assert_array_equal(parts,full)
        changed=p.copy(); changed[200:]=0
        np.testing.assert_array_equal(smooth_probabilities(changed,2)[0][:200],full[:200])
        self.assertEqual(smooth_probabilities(p[200:],2)[0][0],p[200])
    def test_cutoff_excludes_crossing_event(self):
        y=np.array([0,1,1,1,0,0])
        self.assertEqual(event_safe_boundary(y,3),1)
        self.assertEqual(event_safe_boundary(y,4),4)

if __name__=='__main__': unittest.main()
