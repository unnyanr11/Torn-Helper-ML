import unittest
import numpy as np

from strategy_engine import STRATEGY_FAMILIES, run_strategy_competition


class StrategyEngineTests(unittest.TestCase):
    def test_registry_contains_trend_momentum_reversion_breakout_and_volatility(self):
        self.assertTrue({"trend_ema", "trend_macd", "momentum_roc", "momentum_dual",
                         "mean_reversion_bollinger", "mean_reversion_rsi",
                         "breakout_donchian", "volatility_breakout",
                         "bollinger_squeeze"}.issubset(set(STRATEGY_FAMILIES)))

    def test_rejects_bad_or_unaligned_data(self):
        with self.assertRaises(ValueError):
            run_strategy_competition(np.arange(100), np.ones(99), horizon_steps=3, bar_seconds=60)
        with self.assertRaises(ValueError):
            run_strategy_competition(np.arange(100), np.zeros(100), horizon_steps=3, bar_seconds=60)

    def test_outputs_all_strategies_and_explicit_baselines_without_lookahead(self):
        rng = np.random.default_rng(17)
        returns = rng.normal(0.0003, 0.004, 800)
        prices = 100 * np.exp(np.cumsum(returns))
        times = np.arange(len(prices), dtype=np.int64) * 60
        result = run_strategy_competition(times, prices, horizon_steps=12,
            bar_seconds=60, min_trades=1, minimum_net_return=0.0)
        self.assertEqual(result["status"], "completed")
        for name in STRATEGY_FAMILIES:
            self.assertIn(name, result["strategies"])
            for trade in result["strategies"][name]["trades"]:
                self.assertGreater(trade["entry_index"], trade["origin_time"] // 60)
                self.assertGreater(trade["exit_index"], trade["entry_index"])
                self.assertGreaterEqual(trade["exit_time"] - trade["entry_time"], 0)
        self.assertIn("baseline_buy_hold", result["strategies"])
        self.assertIn("baseline_cash", result["strategies"])

    def test_rejects_gaps_in_the_execution_path_instead_of_filling_them(self):
        rng = np.random.default_rng(5)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, .003, 500)))
        times = np.arange(len(prices), dtype=np.int64) * 60
        times[250:] += 3600
        result = run_strategy_competition(times, prices, horizon_steps=5,
            bar_seconds=60, min_trades=1)
        for item in result["strategies"].values():
            for trade in item["trades"]:
                self.assertFalse(np.any(np.diff(times[trade["entry_index"]:trade["exit_index"] + 1]) > 90))

    def test_fixed_profit_threshold_does_not_gate_research_eligibility(self):
        rng = np.random.default_rng(21)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, .002, 1200)))
        times = np.arange(len(prices), dtype=np.int64) * 60
        baseline = run_strategy_competition(times, prices, horizon_steps=8,
            bar_seconds=60, fee_bps=10, slippage_bps=5,
            minimum_net_return=0.0, min_trades=1)
        higher_hurdle = run_strategy_competition(times, prices, horizon_steps=8,
            bar_seconds=60, fee_bps=10, slippage_bps=5,
            minimum_net_return=.002, min_trades=1)
        for name in STRATEGY_FAMILIES:
            self.assertEqual(
                baseline["strategies"][name]["metrics"]["eligible"],
                higher_hurdle["strategies"][name]["metrics"]["eligible"])
        self.assertAlmostEqual(baseline["config"]["round_trip_cost"], .002)



if __name__ == "__main__":
    unittest.main()
