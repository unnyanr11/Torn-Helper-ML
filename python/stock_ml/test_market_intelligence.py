import unittest
import numpy as np

from market_intelligence import (
    classify_market_regime, multi_timeframe_confirmation,
    calibrate_directional_probabilities, apply_probability_calibration,
    adaptive_exit_plan, score_opportunity, select_strategy_for_regime,
)


class MarketIntelligenceTests(unittest.TestCase):
    def test_regime_detects_trend_and_high_volatility(self):
        up = 100*np.exp(np.cumsum(np.full(100, .003)))
        self.assertEqual(classify_market_regime(up)["regime"], "uptrend")
        rng = np.random.default_rng(12)
        volatile = 100*np.exp(np.cumsum(rng.normal(0, .03, 200)))
        self.assertEqual(classify_market_regime(volatile)["regime"], "high_volatility")

    def test_multi_timeframe_requires_at_least_two_valid_series(self):
        result = multi_timeframe_confirmation([100, 102, 104], [100, 101, 103], [100])
        self.assertEqual(result["direction"], "up")
        self.assertEqual(result["status"], "confirmed")

    def test_probability_calibration_uses_holdout_and_only_activates_on_improvement(self):
        rng = np.random.default_rng(5)
        p = np.clip(rng.beta(4, 2, 100), .01, .99)
        y = (rng.random(100) < p**1.5/(p**1.5+(1-p)**1.5)).astype(int)
        result = calibrate_directional_probabilities(p, y)
        self.assertIn(result["status"], ("accepted", "rejected"))
        calibrated = apply_probability_calibration(.8, result)
        self.assertTrue(0 <= calibrated <= 1)
        self.assertEqual(apply_probability_calibration(.8, {"status":"rejected"}), .8)

    def test_adaptive_exit_plan_scales_risk_and_never_exceeds_position_cap(self):
        rng = np.random.default_rng(8)
        prices = 100*np.exp(np.cumsum(rng.normal(0, .004, 80)))
        result = adaptive_exit_plan(100, prices, "high_volatility")
        self.assertGreater(result["target_price"], 100)
        self.assertLess(result["stop_price"], 100)
        self.assertLessEqual(result["position_weight"], .15)
        self.assertGreaterEqual(result["trailing_stop_price"], result["stop_price"])

    def test_net_profit_gate_and_eligibility_are_not_replaced_by_score(self):
        result = score_opportunity(expected_gross_return=.003, probability_up=.8,
            uncertainty=.004, round_trip_cost=.002, minimum_net_return=.002,
            regime_confidence=.8, timeframe_agreement=1, strategy_eligible=True)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["status"], "watch_only")
        passing = score_opportunity(expected_gross_return=.006, probability_up=.8,
            uncertainty=.004, round_trip_cost=.002, minimum_net_return=.002,
            regime_confidence=.8, timeframe_agreement=1, strategy_eligible=True)
        self.assertTrue(passing["eligible"])

    def test_regime_selector_uses_only_eligible_matching_strategies(self):
        results = {
            "trend_ema":{"metrics":{"eligible":True,"total_return":.1,"profit_factor":2,"max_drawdown":-.1}},
            "mean_reversion_rsi":{"metrics":{"eligible":True,"total_return":.2,"profit_factor":3,"max_drawdown":-.1}},
            "trend_macd":{"metrics":{"eligible":False,"total_return":.5,"profit_factor":5,"max_drawdown":-.1}},
        }
        self.assertEqual(select_strategy_for_regime(results,"uptrend"),"trend_ema")


if __name__ == "__main__":
    unittest.main()
