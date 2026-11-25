"""Regression tests for the live/backtest net-profit signal gate."""
import unittest

from realtime_model import (
    DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, DEFAULT_MIN_EXPECTED_NET_RETURN,
    required_gross_return, signal_for_return, make_result,
)


class LiveSignalCostGateTests(unittest.TestCase):
    def test_default_short_term_gate_matches_backtest_costs_and_01_net(self):
        # 10 bps fee + 5 bps slippage on each side = 30 bps round trip.
        self.assertAlmostEqual(required_gross_return(), 0.004)
        self.assertEqual(DEFAULT_MIN_EXPECTED_NET_RETURN, 0.001)
        self.assertEqual(DEFAULT_FEE_BPS, 10.0)
        self.assertEqual(DEFAULT_SLIPPAGE_BPS, 5.0)

    def test_01_and_02_percent_net_thresholds_are_configurable(self):
        self.assertAlmostEqual(required_gross_return(minimum_net_return=0.001), 0.004)
        self.assertAlmostEqual(required_gross_return(minimum_net_return=0.002), 0.005)
        self.assertEqual(signal_for_return(.0051, .001, True, minimum_net_return=.001), 'Buy')
        self.assertEqual(signal_for_return(.0051, .001, True, minimum_net_return=.002), 'Wait')

    def test_buy_waits_below_net_threshold_after_uncertainty(self):
        self.assertEqual(signal_for_return(.0045, .001, True), 'Wait')
        self.assertEqual(signal_for_return(.0051, .001, True), 'Buy')

    def test_sell_uses_symmetric_cost_and_uncertainty_margin(self):
        self.assertEqual(signal_for_return(-.0045, .001, True), 'Wait')
        self.assertEqual(signal_for_return(-.0051, .001, True), 'Sell')

    def test_fee_and_slippage_changes_raise_required_gross_return(self):
        self.assertAlmostEqual(required_gross_return(), .004)
        self.assertAlmostEqual(required_gross_return(fee_bps=15, slippage_bps=10), .006)
        self.assertEqual(signal_for_return(.0051, .001, True, fee_bps=15, slippage_bps=10), 'Wait')

    def test_unapproved_model_never_emits_trade_signal(self):
        self.assertEqual(signal_for_return(.05, .001, False), 'Wait')

    def test_negative_or_invalid_cost_inputs_are_rejected(self):
        for kwargs in ({'fee_bps': -1}, {'slippage_bps': 501},
                       {'minimum_net_return': -0.001},
                       {'minimum_net_return': float('nan')}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    required_gross_return(**kwargs)

    def test_make_result_accepts_per_request_net_threshold(self):
        model = {'status': 'ready',
            'parameters': {'mean': [0.0], 'scale': [1.0], 'coef': [0.0], 'intercept': .0051},
            'metrics': {'error_q90': .001, 'approved': True}, 'model_id': 'test-model'}
        low = make_result(model, None, [0.0], True, minimum_net_return=.001)
        high = make_result(model, None, [0.0], True, minimum_net_return=.002)
        self.assertEqual(low['signal'], 'Buy')
        self.assertEqual(high['signal'], 'Wait')


if __name__ == '__main__':
    unittest.main()
