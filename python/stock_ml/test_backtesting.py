import unittest
import numpy as np
from backtesting import run_backtest, calibration_from_backtest

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

if __name__=='__main__': unittest.main()
