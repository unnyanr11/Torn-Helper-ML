import unittest

from algorithm_registry import (
    ALGORITHM_REGISTRY,
    algorithm_id_for_family,
    family_for_algorithm_id,
)


class AlgorithmRegistryTests(unittest.TestCase):
    def test_algorithm_ids_are_stable_and_cover_every_backtest_family(self):
        self.assertEqual(ALGORITHM_REGISTRY, {
            1: "ridge",
            2: "elastic_net",
            3: "random_forest",
            4: "extra_trees",
            5: "hist_gradient_boosting",
            6: "ensemble",
        })
        self.assertEqual(len(set(ALGORITHM_REGISTRY.values())), 6)

    def test_family_and_number_round_trip(self):
        for algorithm_id, family in ALGORITHM_REGISTRY.items():
            self.assertEqual(algorithm_id_for_family(family), algorithm_id)
            self.assertEqual(family_for_algorithm_id(algorithm_id), family)

    def test_unknown_algorithm_ids_and_families_are_rejected(self):
        with self.assertRaises(ValueError):
            family_for_algorithm_id(99)
        with self.assertRaises(ValueError):
            algorithm_id_for_family("unregistered")


if __name__ == "__main__":
    unittest.main()
