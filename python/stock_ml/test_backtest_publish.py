import unittest
from backtest_publish import choose_winner


class BacktestPublishTests(unittest.TestCase):
    def row(self,family,mae,baseline=.01,folds=40,directional=.6,rmse=.012):
        return dict(model_family=family,mae=mae,baseline_mae=baseline,folds=folds,
                    directional_accuracy=directional,rmse=rmse)

    def test_selects_lowest_mae_that_beats_no_change_by_two_percent(self):
        rows=[self.row("ridge",.0098),self.row("extra_trees",.007),
              self.row("random_forest",.0101)]
        winner=choose_winner(rows)
        self.assertEqual(winner[3],"extra_trees")

    def test_directional_accuracy_breaks_equal_mae_ties(self):
        winner=choose_winner([self.row("ridge",.007,directional=.55),
                              self.row("elastic_net",.007,directional=.7)])
        self.assertEqual(winner[3],"elastic_net")

    def test_rejects_insufficient_folds(self):
        self.assertIsNone(choose_winner([self.row("ridge",.005,folds=29)]))

    def test_rejects_model_that_does_not_beat_no_change_by_two_percent(self):
        self.assertIsNone(choose_winner([self.row("ridge",.0099)]))

    def test_rejects_unknown_model_family(self):
        self.assertIsNone(choose_winner([self.row("unregistered",.005)]))


if __name__=="__main__":
    unittest.main()
