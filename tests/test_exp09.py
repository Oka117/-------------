import unittest
import numpy as np
import pandas as pd
from experiments.exp09 import partitions, calibration_block, assert_complete
from experiments.diagnostics import diagnose
from metrics import event_weights


class ForwardProtocolTests(unittest.TestCase):
    def test_cross_month_event_and_gap(self):
        f=pd.DataFrame({'timestamp':pd.date_range('2024-01-01','2024-10-04 23:40',freq='20min')})
        y=np.zeros(len(f),dtype=np.int8)
        a=int(f.timestamp.searchsorted(pd.Timestamp('2024-03-30')))
        b=int(f.timestamp.searchsorted(pd.Timestamp('2024-04-03')))
        y[a:b]=1
        boundaries,tasks=partitions(f,y)
        self.assertEqual(boundaries[1]['index'],a)
        for t in tasks:
            if t['status']=='available':
                self.assertGreaterEqual(t['cal_start']-t['fit_end'],72)
                self.assertEqual(t['cal_end'],t['eval_start'])
                for left,right in [(0,t['fit_end']),(t['cal_start'],t['cal_end']),(t['eval_start'],t['eval_end'])]:
                    assert_complete(y,left,right)
        self.assertEqual(tasks[0]['eval_end'],int(f.timestamp.searchsorted(pd.Timestamp('2024-05-01'))))

    def test_merged_boundaries_report_missing_month(self):
        f=pd.DataFrame({'timestamp':pd.date_range('2024-01-01','2024-10-04 23:40',freq='20min')})
        y=np.zeros(len(f),dtype=np.int8)
        y[(f.timestamp>='2024-03-28')&(f.timestamp<'2024-05-03')]=1
        boundaries,tasks=partitions(f,y)
        self.assertTrue(boundaries[2]['merged_into_earlier'])
        self.assertEqual(tasks[1]['status'],'unavailable')
        self.assertEqual(tasks[0]['eval_end'],boundaries[3]['index'])

    def test_common_pool_drops_crossing_event_and_future_labels(self):
        f=pd.DataFrame(dict(timestamp=pd.date_range('2024-04-01',periods=8,freq='20min'),
            label=[0,0,1,1,1,0,0,0],probability=np.arange(8)/8))
        b,audit=calibration_block(f,True,'2024-04-01 01:00')
        self.assertEqual(audit['retained_rows'],2)
        np.testing.assert_array_equal(b['y'],[0,0])
        changed=f.copy(); changed.loc[5:,'label']=1
        c,_=calibration_block(changed,True,'2024-04-01 01:00')
        np.testing.assert_array_equal(b['y'],c['y'])
        _,audit=calibration_block(f,False,'2024-04-01 01:00')
        self.assertFalse(audit['included'])

    def test_missed_event_is_null_and_short_window_uses_actual_length(self):
        f=pd.DataFrame(dict(timestamp=['a','b','c','d'],label=[0,1,1,0],probability=[0.,0.,0.,0.]))
        detail,_=diagnose(f,.5)
        self.assertIsNone(detail['events'][0]['delay_minutes'])
        self.assertTrue(detail['events'][0]['missed_event'])
        f.loc[2,'probability']=1.
        detail,_=diagnose(f,.5)
        self.assertEqual(detail['events'][0]['recall_3h'],.5)
        self.assertEqual(detail['events'][0]['delay_minutes'],20)


if __name__=='__main__': unittest.main()
