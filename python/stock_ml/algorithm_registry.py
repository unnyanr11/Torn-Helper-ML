"""Stable numbered identifiers for short-term prediction algorithms.

IDs are persisted with every published model and prediction. Never reorder or
reuse an ID; append new algorithms with a new identifier.
"""
from __future__ import annotations

ALGORITHM_REGISTRY = {
    1: "ridge",
    2: "elastic_net",
    3: "random_forest",
    4: "extra_trees",
    5: "hist_gradient_boosting",
    6: "ensemble",
}
FAMILY_TO_ALGORITHM_ID = {family: algorithm_id
                          for algorithm_id, family in ALGORITHM_REGISTRY.items()}


def algorithm_id_for_family(family: str) -> int:
    try:
        return FAMILY_TO_ALGORITHM_ID[family]
    except KeyError as exc:
        raise ValueError(f"Unregistered prediction algorithm: {family}") from exc


def family_for_algorithm_id(algorithm_id: int) -> str:
    try:
        return ALGORITHM_REGISTRY[int(algorithm_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown prediction algorithm ID: {algorithm_id}") from exc
