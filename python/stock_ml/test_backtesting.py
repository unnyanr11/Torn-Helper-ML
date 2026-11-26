import unittest
import numpy as np
from backtesting import run_backtest, run_backtest_competition, calibration_from_backtest, MODEL_FAMILIES, _family_predict
from realtime_model import train_range, RIDGE_ALPHAS, COMBINATIONS, feature_matrix, infer_exported
from backtest_runner import _align_model_results
from backtesting import fit_live_model

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
        self.assertEqual(len(COMBINATIONS),36)
        self.assertEqual(len(set(COMBINATIONS)),36)
        self.assertEqual({style for style, _, _ in COMBINATIONS}, {'short'})
        self.assertEqual(
            {window for _, window, _ in COMBINATIONS},
            {720, 1080, 1440, 4320, 10080, 43200},
        )
        self.assertGreaterEqual(min(window for _, window, _ in COMBINATIONS), 720)

    def test_competition_reuses_feature_preparation_and_matches_single_family_results(self):
        from unittest.mock import patch
        import backtesting

        rng=np.random.default_rng(231)
        times=np.arange(0,20000,dtype=np.int64)
        prices=100*np.exp(np.cumsum(rng.normal(0.00001,0.0002,len(times))))
        with patch('backtesting.prepare_series', wraps=backtesting.prepare_series) as prepare:
            grouped=run_backtest_competition(
                times,prices,window_minutes=60,horizon_minutes=15,max_folds=3,
                min_train_rows=200,model_families=('ridge','elastic_net'),
            )
            # The full causal feature matrix is built once and reused by all folds.
            self.assertEqual(prepare.call_count,1)

        for family in ('ridge','elastic_net'):
            individual=run_backtest(
                times,prices,window_minutes=60,horizon_minutes=15,max_folds=3,
                min_train_rows=200,model_family=family,
            )
            self.assertEqual(
                [p['origin_minute'] for p in grouped[family]['predictions']],
                [p['origin_minute'] for p in individual['predictions']],
            )
            np.testing.assert_allclose(
                [p['predicted_return'] for p in grouped[family]['predictions']],
                [p['predicted_return'] for p in individual['predictions']],
                rtol=0,atol=1e-12,
            )
            np.testing.assert_allclose(
                [p['actual_return'] for p in grouped[family]['predictions']],
                [p['actual_return'] for p in individual['predictions']],
                rtol=0,atol=1e-12,
            )

    def test_ensemble_reuses_base_model_fits(self):
        from unittest.mock import patch
        import backtesting

        rng=np.random.default_rng(232)
        times=np.arange(0,20000,dtype=np.int64)
        prices=100*np.exp(np.cumsum(rng.normal(0.00001,0.0002,len(times))))
        original_fit=backtesting._fit_predict
        with patch('backtesting._fit_predict', wraps=original_fit) as fit:
            results=run_backtest_competition(
                times,prices,window_minutes=60,horizon_minutes=15,max_folds=3,
                min_train_rows=200,model_families=MODEL_FAMILIES,
            )
        # Five base families × validation/final fit per eligible fold. The ensemble
        # must reuse those predictions rather than fitting the five models again.
        fold_count=len(results['ridge']['predictions'])
        self.assertGreaterEqual(fold_count,1)
        self.assertEqual(fit.call_count,5*2*fold_count)
        self.assertEqual(set(results),set(MODEL_FAMILIES))
        self.assertEqual(
            [p['origin_minute'] for p in results['ensemble']['predictions']],
            [p['origin_minute'] for p in results['ridge']['predictions']],
        )

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

    def test_exported_live_models_support_every_backtest_family(self):
        rng=np.random.default_rng(91)
        times=np.arange(0,9000,dtype=np.int64)
        prices=100*np.exp(np.cumsum(rng.normal(0.00001,0.0003,len(times))))
        origin=int(times[-1])
        x,valid=feature_matrix(times,prices,[origin],60)
        self.assertTrue(bool(valid[0]))
        for family in MODEL_FAMILIES:
            model=fit_live_model(times,prices,window_minutes=60,horizon_minutes=15,
                                 model_family=family)
            self.assertEqual(model['status'],'ready')
            self.assertEqual(model['model_family'],family)
            from algorithm_registry import algorithm_id_for_family
            self.assertEqual(model['algorithm_id'],algorithm_id_for_family(family))
            prediction=infer_exported(model['parameters'],x[0])
            self.assertTrue(np.isfinite(prediction),family)

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


    def test_live_fit_reuses_cached_window_features_without_changing_model(self):
        from unittest.mock import patch
        import backtesting
        rng = np.random.default_rng(811)
        times = np.arange(0, 12000, dtype=np.int64)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.00001, 0.0003, len(times))))
        observed = np.ones(len(times), dtype=bool)
        origins = times[times % 15 == 0]
        features, valid = feature_matrix(times, prices, origins, 60)
        cache = (origins, features, valid)
        direct = fit_live_model(times, prices, window_minutes=60, horizon_minutes=30,
                                model_family='ridge', observed_mask=observed)
        with patch('backtesting.feature_matrix', wraps=backtesting.feature_matrix) as build:
            cached = fit_live_model(times, prices, window_minutes=60, horizon_minutes=30,
                                    model_family='ridge', observed_mask=observed,
                                    feature_data=cache)
            self.assertEqual(build.call_count, 0)
        self.assertEqual(direct['training_rows'], cached['training_rows'])
        for key in ('mean', 'scale', 'coef'):
            np.testing.assert_allclose(direct['parameters'][key], cached['parameters'][key],
                                       rtol=0, atol=0)
        self.assertEqual(direct['parameters']['intercept'], cached['parameters']['intercept'])

    def test_live_fit_rejects_misaligned_cached_feature_origins(self):
        rng = np.random.default_rng(812)
        times = np.arange(0, 9000, dtype=np.int64)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.0002, len(times))))
        origins = times[times % 15 == 0]
        features, valid = feature_matrix(times, prices, origins, 60)
        with self.assertRaisesRegex(ValueError, 'Cached feature data'):
            fit_live_model(times, prices, window_minutes=60, horizon_minutes=30,
                           model_family='ridge', feature_data=(origins[1:], features, valid))

if __name__=='__main__': unittest.main()
