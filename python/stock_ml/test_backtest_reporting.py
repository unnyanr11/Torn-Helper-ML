import unittest
from datetime import datetime, timezone
from backtest_reporting import rank_rows

class BacktestReportingTests(unittest.TestCase):
    def row(self, stock, family, mae, baseline, n=40, window=60, horizon=60):
        return (stock,'short',window,horizon,family,'00000000-0000-0000-0000-000000000001',
                datetime.now(timezone.utc),n,mae,mae*1.2,baseline,.6,0.0,.03,10,.6,-.04,1.5)

    def test_requires_minimum_samples_and_two_percent_improvement(self):
        report=rank_rows([self.row(1,'ridge',.009,.01,n=29),
                          self.row(1,'elastic_net',.0099,.01)])
        self.assertEqual(report['eligible_runs'],0)

    def test_ranks_eligible_models_and_selects_best_per_stock(self):
        report=rank_rows([self.row(1,'ridge',.009,.01),
                          self.row(1,'extra_trees',.007,.01),
                          self.row(2,'ridge',.012,.01)])
        self.assertEqual(report['eligible_runs'],2)
        self.assertEqual(report['best_model_by_stock'][0]['model_family'],'extra_trees')
        self.assertEqual(report['best_model_by_stock'][0]['stock_id'],1)


    def test_rejects_high_return_model_with_excessive_drawdown(self):
        risky=self.row(1,'random_forest',.007,.01)
        risky=risky[:-2]+(-.30,2.0)
        safer=self.row(1,'extra_trees',.008,.01)
        safer=safer[:-2]+(-.08,1.6)
        report=rank_rows([risky,safer])
        self.assertEqual(report['best_model_by_stock'][0]['model_family'],'extra_trees')
        rejected=next(x for x in report['rankings'] if x['model_family']=='random_forest')
        self.assertFalse(rejected['eligible'])

    def test_rejects_no_trade_models(self):
        item=self.row(1,'no_trades',.007,.01)
        item=item[:13]+(0.0,0,None,-.0,None)
        report=rank_rows([item])
        self.assertEqual(report['eligible_runs'],0)

    def test_reports_risk_adjusted_score(self):
        report=rank_rows([self.row(1,'ridge',.007,.01)])
        self.assertAlmostEqual(report['rankings'][0]['risk_adjusted_score'],.03/.04)

    def test_groups_best_by_stock_and_configuration(self):
        report=rank_rows([self.row(1,'ridge',.009,.01),
                          self.row(1,'extra_trees',.007,.01)])
        self.assertEqual(len(report['best_model_by_configuration']),1)

if __name__=='__main__': unittest.main()
