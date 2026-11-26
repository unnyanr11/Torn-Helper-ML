import unittest
import numpy as np
from backtesting import run_backtest, calibration_from_backtest, MODEL_FAMILIES, _family_predict
from realtime_model import train_range, RIDGE_ALPHAS, COMBINATIONS
from backtest_runner import _align_model_results

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

    def test_backtest_api_has_no_trading_cost_or_portfolio_parameters(self):
        import inspect
        parameters=set(inspect.signature(run_backtest).parameters)
        self.assertFalse({'fee_bps','slippage_bps','minimum_net_return','initial_capital','trade_on'} & parameters)

    def test_backtest_reports_prediction_accuracy_without_trade_simulation(self):
        rng=np.random.default_rng(23)
        times=np.arange(0,12000,dtype=np.int64)
        prices=100*np.exp(np.cumsum(rng.normal(0.00001,0.0002,len(times))))
        result=run_backtest(times,prices,window_minutes=60,horizon_minutes=15,
                            max_folds=3,min_train_rows=200,model_family='ridge')
        self.assertEqual(result['status'],'completed')
        self.assertIn('directional_accuracy',result['metrics'])
        self.assertNotIn('trades',result)
        self.assertNotIn('total_return',result['metrics'])
        self.assertNotIn('fee_bps',result['config'])
        self.assertNotIn('slippage_bps',result['config'])


    def test_model_comparison_aligns_identical_walk_forward_origins(self):
        def row(origin, predicted, actual):
            return dict(origin_minute=origin, predicted_return=predicted,
                        actual_return=actual, raw_return=predicted)
        results = {
            'ridge': dict(predictions=[row(10, .01, .02), row(20, -.01, -.02), row(30, .02, .01)],
                          metrics={}, config={}),
            'random_forest': dict(predictions=[row(10, .02, .02), row(20, -.02, -.02)],
                                  metrics={}, config={}),
        }
        aligned = _align_model_results(results)
        self.assertEqual(
            [p['origin_minute'] for p in aligned['ridge']['predictions']],
            [10, 20],
        )
        self.assertEqual(
            [p['origin_minute'] for p in aligned['random_forest']['predictions']],
            [10, 20],
        )
        self.assertEqual(aligned['ridge']['metrics']['folds'], 2)
        self.assertEqual(aligned['random_forest']['metrics']['folds'], 2)
        self.assertEqual(
            aligned['ridge']['metrics']['comparison_fold_policy'],
            'intersection_of_all_model_family_origins',
        )

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
