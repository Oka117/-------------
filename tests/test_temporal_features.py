import unittest
import numpy as np
import pandas as pd
from temporal_features import WINDOWS,build_features,inference_features,fit_recipe


def recipe(kind):
    return dict(raw_features=['feature_0'],selected=['feature_0'],kind=kind,windows=WINDOWS[kind],epsilon={'feature_0':1e-6},history_points=72)


class TemporalTests(unittest.TestCase):
    def test_values_and_warmup(self):
        raw=pd.DataFrame({'feature_0':np.arange(100,dtype=np.float32)})
        a=build_features(raw,recipe('mean_deviation'))
        self.assertTrue(np.isnan(a.iloc[2,1]))
        self.assertEqual(a.iloc[3,1],2)
        b=build_features(raw,recipe('lag_difference'))
        self.assertEqual(b.iloc[20,1],1)
        c=build_features(raw,recipe('history_std'))
        self.assertAlmostEqual(c.iloc[3,1],np.std([0,1,2]),places=6)
        d=build_features(raw,recipe('standardized_deviation'))
        self.assertAlmostEqual(d.iloc[18,1],9.5/np.std(np.arange(18)),places=6)

    def test_future_causality_and_chunks(self):
        rng=np.random.default_rng(12)
        raw=pd.DataFrame({'feature_0':rng.normal(size=250).astype('float32')})
        for kind in WINDOWS:
            r=recipe(kind)
            full=build_features(raw,r)
            altered=raw.copy();altered.iloc[150:]=1000
            pd.testing.assert_frame_equal(build_features(altered,r).iloc[:150],full.iloc[:150])
            chunk=build_features(raw.iloc[78:],r).iloc[72:].reset_index(drop=True)
            np.testing.assert_allclose(chunk,full.iloc[150:],rtol=1e-6,atol=1e-6,equal_nan=True)

    def test_train_test_seam_and_gap_reset(self):
        df=pd.DataFrame({'timestamp':pd.date_range('2024-01-01',periods=150,freq='20min'), 'feature_0':np.arange(150,dtype='float32')})
        for kind in WINDOWS:
            r=recipe(kind)
            expected=build_features(df[['feature_0']],r).iloc[100:].reset_index(drop=True)
            got=inference_features(df.iloc[100:],df.iloc[28:100],r)
            pd.testing.assert_frame_equal(got,expected)
            gap=df.iloc[100:].copy();gap['timestamp']+=pd.Timedelta(days=1)
            self.assertTrue(inference_features(gap,df.iloc[28:100],r).iloc[0,1:].isna().all())

    def test_long_constant_history_exact_std_chunk_parity(self):
        rng=np.random.default_rng(7)
        values=np.r_[rng.normal(size=500), np.full(1000, .12345678)].astype('float32')
        raw=pd.DataFrame({'feature_0':values})
        for kind in ['history_std','standardized_deviation']:
            r=recipe(kind)
            full=build_features(raw,r).iloc[800:].reset_index(drop=True)
            chunk=build_features(raw.iloc[728:],r).iloc[72:].reset_index(drop=True)
            pd.testing.assert_frame_equal(full,chunk,check_exact=True)
            self.assertTrue((full.iloc[:,1:]==0).all().all())

    def test_constant_features_and_bad_raw(self):
        raw=pd.DataFrame({'feature_0':np.ones(100,dtype='float32')})
        r=fit_recipe(raw,np.zeros(100),'standardized_deviation')
        self.assertEqual(r['selection_source'],'first_30_schema_columns_single_class')
        self.assertEqual(r['epsilon']['feature_0'],1e-6)
        self.assertTrue((build_features(raw,r).iloc[72:,1:]==0).all().all())
        raw.iloc[0,0]=np.nan
        with self.assertRaises(ValueError):build_features(raw,r)


if __name__=='__main__':unittest.main()
