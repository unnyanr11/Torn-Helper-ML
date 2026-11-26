import unittest
import numpy as np

from clean_series import reconstruct_short_gaps, contiguous_segments


class CleanSeriesTests(unittest.TestCase):
    def test_one_missing_minute_is_average_and_marked_synthetic(self):
        t, p, synthetic, methods = reconstruct_short_gaps([10, 12], [100, 104])
        np.testing.assert_array_equal(t, [10, 11, 12])
        np.testing.assert_allclose(p, [100, 102, 104])
        np.testing.assert_array_equal(synthetic, [False, True, False])
        self.assertEqual(methods, {11: "linear_average"})

    def test_two_missing_minutes_are_interpolated(self):
        t, p, synthetic, methods = reconstruct_short_gaps([10, 13], [100, 106])
        np.testing.assert_array_equal(t, [10, 11, 12, 13])
        np.testing.assert_allclose(p, [100, 102, 104, 106])
        self.assertEqual(synthetic.sum(), 2)
        self.assertEqual(methods, {11: "linear_average"})

    def test_three_to_five_missing_minutes_use_slope_bridge(self):
        t, p, synthetic, methods = reconstruct_short_gaps([10, 15], [100, 110])
        np.testing.assert_array_equal(t, np.arange(10, 16))
        np.testing.assert_allclose(p, np.arange(100, 111, 2))
        self.assertEqual(synthetic.sum(), 4)
        self.assertEqual(methods, {11: "local_slope_bridge"})

    def test_gap_over_five_is_not_fabricated_and_splits_segments(self):
        t, p, synthetic, _ = reconstruct_short_gaps([10, 17], [100, 107])
        np.testing.assert_array_equal(t, [10, 17])
        self.assertFalse(synthetic.any())
        segments = contiguous_segments(t, p, synthetic)
        self.assertEqual([len(x[0]) for x in segments], [1, 1])

    def test_duplicate_or_invalid_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct_short_gaps([10, 10], [100, 101])
        with self.assertRaises(ValueError):
            reconstruct_short_gaps([10, 11], [100, 0])


if __name__ == "__main__":
    unittest.main()
