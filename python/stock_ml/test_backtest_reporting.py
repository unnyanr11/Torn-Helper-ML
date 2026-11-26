import unittest
from datetime import datetime, timezone
from backtest_reporting import rank_rows


class BacktestReportingTests(unittest.TestCase):
    def row(self, stock, family, mae, baseline, n=40, window=60, horizon=60):
        return (stock,'short',window,horizon,family,'00000000-0000-0000-0000-000000000001',
                datetime.now(timezone.utc),n,mae,mae*1.2,baseline,.6,0.0,
                None,None,None,None,None)

    def test_requires_minimum_samples_and_two_percent_accuracy_improvement(self):
        report=rank_rows([self.row(1,'ridge',.007,.01,n=29),
                          self.row(1,'elastic_net',.0099,.01)])
        self.assertEqual(report['eligible_runs'],0)

    def test_ranks_by_prediction_error_and_selects_best_per_stock(self):
        report=rank_rows([self.row(1,'ridge',.009,.01),
                          self.row(1,'extra_trees',.007,.01),
                          self.row(2,'ridge',.012,.01)])
        self.assertEqual(report['eligible_runs'],2)
        self.assertEqual(report['best_model_by_stock'][0]['model_family'],'extra_trees')
        self.assertEqual(report['best_model_by_stock'][0]['stock_id'],1)

    def test_trading_profit_and_drawdown_fields_do_not_gate_accuracy_backtests(self):
        report=rank_rows([self.row(1,'random_forest',.007,.01)])
        self.assertEqual(report['eligible_runs'],1)
        self.assertIsNone(report['rankings'][0]['total_return'])
        self.assertIsNone(report['rankings'][0]['max_drawdown'])
        self.assertIn('MAE',report['ranking_method'])

    def test_groups_best_by_stock_and_configuration(self):
        report=rank_rows([self.row(1,'ridge',.009,.01),
                          self.row(1,'extra_trees',.007,.01)])
        self.assertEqual(len(report['best_model_by_configuration']),1)


if __name__=='__main__':
    unittest.main()
