import unittest
import numpy as np
from backtesting import run_backtest, calibration_from_backtest, MODEL_FAMILIES, _family_predict
from realtime_model import (train_range, RIDGE_ALPHAS, COMBINATIONS, DEFAULT_MIN_EXPECTED_NET_RETURN, required_gross_return)

class BacktestTests(unittest.TestCase):
    def test_model_family_matrix_includes_baselines_trees_and_ensemble(self):
        self.assertEqual(set(MODEL_FAMILIES), {'ridge','elastic_net','random_forest','extra_trees','hist_gradient_boosting','ensemble'})

    def test_each_model_family_returns_finite_predictions(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(80,8)); y=rng.normal(0,.01,size=80); x_eval=rng.normal(size=(5,8))
        for family in MODEL_FAMILIES:
            prediction=_family_predict(family,x,y,x_eval)
            self.assertEqual(prediction.shape,(5,))
            self.assertTrue(np.all(np.isfinite(prediction)))

    def test_backtest_matrix_covers_all_registered_live_combinations(self):
        self.assertEqual(len(COMBINATIONS),72)
        self.assertEqual(len(set(COMBINATIONS)),72)
        self.assertEqual({style for style, _, _ in COMBINATIONS}, {'short'})

    def test_backtest_rejects_long_term_style(self):
        with self.assertRaisesRegex(ValueError, 'Invalid model style'):
            run_backtest(np.arange(1000), np.ones(1000), window_minutes=60,
                         horizon_minutes=60, style='long')

    def test_rejects_invalid_costs(self):
        with self.assertRaises(ValueError):
            run_backtest(np.arange(1000),np.ones(1000),window_minutes=60,horizon_minutes=60,fee_bps=999)

    def test_backtest_uses_same_default_net_threshold_and_round_trip_costs(self):
        self.assertAlmostEqual(required_gross_return(), .004)
        self.assertEqual(DEFAULT_MIN_EXPECTED_NET_RETURN, .001)
        with self.assertRaises(ValueError):
            run_backtest(np.arange(1000),np.ones(1000),window_minutes=60,
                         horizon_minutes=60,minimum_net_return=-.001)

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
        self.assertIn(result['metrics']['selected_family'],('ridge','elastic_net','ridge_elastic_blend'))
        if result['metrics']['selected_family']=='ridge':
            self.assertIn(result['metrics']['selected_alpha'],RIDGE_ALPHAS)
        else:
            self.assertGreaterEqual(result['metrics']['selected_alpha'],0.0)
        self.assertGreaterEqual(result['metrics']['selection_rows'],30)
        self.assertGreaterEqual(result['metrics']['validation_rows'],50)

if __name__=='__main__': unittest.main()
