import unittest
import numpy as np
from backtesting import run_backtest, calibration_from_backtest
from realtime_model import train_range, RIDGE_ALPHAS

class BacktestTests(unittest.TestCase):
    def test_rejects_invalid_costs(self):
        with self.assertRaises(ValueError):
            run_backtest(np.arange(1000),np.ones(1000),window_minutes=60,horizon_minutes=60,fee_bps=999)

    def test_calibration_needs_minimum_samples(self):
        result=calibration_from_backtest([dict(raw_return=.01,actual_return=.02,origin_minute=i) for i in range(10)])
        self.assertFalse(result['active'])
        self.assertEqual(result['scale'],1.0)

    def test_calibration_only_promotes_holdout_improvement(self):
        rows=[dict(raw_return=.01,actual_return=.02,origin_minute=i) for i in range(40)]
        result=calibration_from_backtest(rows)
        self.assertTrue(result['active'])
        self.assertGreater(result['holdout_mae_raw'],result['holdout_mae_corrected'])

    def test_nested_model_selection_uses_supported_alpha(self):
        rng=np.random.default_rng(7)
        times=np.arange(0,30000,dtype=np.int64)
        returns=rng.normal(0.00001,0.0005,len(times))
        prices=100*np.exp(np.cumsum(returns))
        result=train_range(times,prices,'short',60,[15])[15]
        self.assertEqual(result['status'],'ready')
        self.assertIn(result['metrics']['selected_alpha'],RIDGE_ALPHAS)
        self.assertGreaterEqual(result['metrics']['selection_rows'],30)
        self.assertGreaterEqual(result['metrics']['validation_rows'],50)

if __name__=='__main__': unittest.main()
