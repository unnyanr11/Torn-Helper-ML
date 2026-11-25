"""Regression tests for the short-term live signal trading-cost gate."""
import unittest

from realtime_model import make_result, MIN_PREDICTED_GROSS_RETURN, MIN_EXPECTED_NET_RETURN, ESTIMATED_TRADING_COST


def model_with_return(predicted_return, uncertainty=0.001, approved=True):
    return {
        "status": "ready",
        "parameters": {
            "mean": [0.0], "scale": [1.0], "coef": [0.0],
            "intercept": predicted_return,
        },
        "metrics": {"error_q90": uncertainty, "approved": approved},
        "model_id": "test-model",
    }


class LiveSignalCostGateTests(unittest.TestCase):
    def test_short_term_default_minimum_net_return_is_point_one_percent(self):
        self.assertAlmostEqual(MIN_EXPECTED_NET_RETURN, 0.001)
        self.assertAlmostEqual(ESTIMATED_TRADING_COST, 0.003)
    def test_buy_requires_round_trip_cost_plus_minimum_net_after_uncertainty(self):
        below = make_result(model_with_return(MIN_PREDICTED_GROSS_RETURN + 0.0005), None, [0.0], True)
        above = make_result(model_with_return(MIN_PREDICTED_GROSS_RETURN + 0.0011), None, [0.0], True)
        self.assertEqual(below["signal"], "Wait")
        self.assertEqual(above["signal"], "Buy")

    def test_sell_requires_symmetric_cost_and_uncertainty_margin(self):
        below = make_result(model_with_return(-MIN_PREDICTED_GROSS_RETURN - 0.0005), None, [0.0], True)
        above = make_result(model_with_return(-MIN_PREDICTED_GROSS_RETURN - 0.0011), None, [0.0], True)
        self.assertEqual(below["signal"], "Wait")
        self.assertEqual(above["signal"], "Sell")

    def test_custom_minimum_net_return_is_applied_independently(self):
        required = 0.001 + 0.002
        below = make_result(model_with_return(required + 0.0005), None, [0.0], True,
                            minimum_net_return=0.002, estimated_trading_cost=0.001)
        above = make_result(model_with_return(required + 0.0011), None, [0.0], True,
                            minimum_net_return=0.002, estimated_trading_cost=0.001)
        self.assertEqual(below['signal'], 'Wait')
        self.assertEqual(above['signal'], 'Buy')

    def test_unapproved_model_never_emits_trade_signal(self):
        result = make_result(model_with_return(0.05, approved=False), None, [0.0], True)
        self.assertEqual(result["signal"], "Wait")


if __name__ == "__main__":
    unittest.main()
