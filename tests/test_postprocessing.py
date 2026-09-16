import unittest
import numpy as np
from postprocessing import postprocess_probabilities as process
from experiments.exp07 import calibrate


class PostprocessingTests(unittest.TestCase):
    def test_immediate_trigger_hold_release(self):
        p=[.4,.8,.5,.3,.2,.7]
        pred,state=process(p,.7,dict(mode='hysteresis',low_threshold=.3))
        np.testing.assert_array_equal(pred,[0,1,1,1,0,1])
        self.assertTrue(state['active'])

    def test_equal_threshold_is_baseline(self):
        p=np.random.default_rng(2).random(200)
        np.testing.assert_array_equal(process(p,.4,dict(mode='hysteresis',low_threshold=.4))[0],p>=.4)

    def test_causal_smoothing_startup(self):
        np.testing.assert_array_equal(process([.8,.2,.2,.9],.5,dict(mode='smoothing',window=3))[0],[1,1,0,0])

    def test_future_independence_and_chunk_parity(self):
        p=np.random.default_rng(7).random(1001)
        for conf in [dict(mode='none'),dict(mode='hysteresis',low_threshold=.2),dict(mode='smoothing',window=2),dict(mode='smoothing',window=3)]:
            whole=process(p,.6,conf)[0]
            altered=p.copy();altered[301:]=1-altered[301:]
            np.testing.assert_array_equal(whole[:301],process(altered,.6,conf)[0][:301])
            state=None;parts=[]
            for i in range(0,len(p),137):
                pp,state=process(p[i:i+137],.6,conf,state);parts.extend(pp)
            np.testing.assert_array_equal(parts,whole)

    def test_state_not_mutated_and_cold_reset(self):
        conf=dict(mode='hysteresis',low_threshold=.2)
        _,state=process([.9],.8,conf)
        self.assertEqual(process([.4],.8,conf,state)[0][0],1)
        self.assertEqual(process([.4],.8,conf)[0][0],0)
        process([0],.8,conf,state)
        self.assertTrue(state['active'])

    def test_invalid_input(self):
        for p in [[float('nan')],[-.1],[1.1],[[.2]]]:
            with self.assertRaises(ValueError): process(p,.5)
        with self.assertRaises(ValueError): process([.2],.5,dict(mode='hysteresis',low_threshold=.6))
        with self.assertRaises(ValueError): process([.2],.5,dict(mode='smoothing',window=0))

    def test_calibration_ignores_ineligible_blocks(self):
        b=dict(y=np.array([0,1,0,1,0]),p=np.array([.1,.8,.1,.9,.1]),eligible=True)
        invalid=dict(y=np.array([1,0,1]),p=np.array([0.,1.,0.]),eligible=False)
        for conf in [dict(mode='none'),dict(mode='hysteresis',ratio=.5),dict(mode='smoothing',window=2)]:
            a=calibrate({'x':[b],'z':[]},conf)
            c=calibrate({'x':[b,invalid],'z':[invalid]},conf)
            self.assertEqual(a,c)


if __name__=='__main__': unittest.main()
