"""Class-based current-action and ordered NEXT metrics.

Inputs are already mapped to official ``(verb_id, noun_id)`` pairs. ``None``
means unavailable or unmapped; an unknown prediction is always an error on an
eligible reference. Text extraction and class mapping belong outside this file.

NEXT is one ordered forecast, not a set of alternatives. It reports both
classic Levenshtein distance and restricted Damerau distance (optimal string
alignment, allowing adjacent transpositions), divided by the fixed horizon.
These are single-forecast diagnostics, not the INSIGHT/Ego4D best-of-five,
twenty-action benchmark score.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ActionPair = tuple[int | None, int | None]
UNKNOWN: ActionPair = (None, None)
COMPONENTS = {"verb": (0,), "noun": (1,), "action": (0, 1)}


def _pair(value: Any, label: str) -> ActionPair:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element verb/noun pair")
    for item in value:
        if item is not None and (type(item) is not int or item < 0):
            raise ValueError(f"{label} class IDs must be nonnegative integers or None")
    return value[0], value[1]


def _pairs(values: Any, label: str) -> list[ActionPair]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of verb/noun pairs")
    return [_pair(value, f"{label}[{index}]") for index, value in enumerate(values)]


def score_action_pairs(
    predictions: Sequence[ActionPair], references: Sequence[ActionPair],
) -> dict[str, dict[str, int | float | None]]:
    """Return independently eligible verb, noun, and verb--noun accuracies.

    Eligibility depends only on whether the reference supplies that component.
    ``mapped_predictions`` counts eligible rows with all required predicted IDs
    present; it does not count correct rows or remove unknowns from denominators.
    """
    predicted = _pairs(predictions, "predictions")
    gold = _pairs(references, "references")
    if len(predicted) != len(gold):
        raise ValueError("predictions and references must contain the same number of examples")
    result = {}
    for name, components in COMPONENTS.items():
        eligible = [(p, g) for p, g in zip(predicted, gold)
                    if all(g[index] is not None for index in components)]
        correct = sum(all(p[index] is not None and p[index] == g[index]
                          for index in components) for p, g in eligible)
        mapped = sum(all(p[index] is not None for index in components)
                     for p, _ in eligible)
        denominator = len(eligible)
        result[name] = {
            "correct": correct,
            "denominator": denominator,
            "rate": correct / denominator if denominator else None,
            "mapped_predictions": mapped,
        }
    return result


def levenshtein_distance(prediction: Sequence[Any], reference: Sequence[Any]) -> int:
    """Unit-cost Levenshtein distance, without an adjacent-transposition edit."""
    previous = list(range(len(reference) + 1))
    for row, predicted in enumerate(prediction, 1):
        current = [row]
        for column, gold in enumerate(reference, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (predicted != gold),
            ))
        previous = current
    return previous[-1]


def damerau_osa_distance(prediction: Sequence[Any], reference: Sequence[Any]) -> int:
    """Restricted Damerau / optimal-string-alignment distance.

    Insertions, deletions, substitutions, and adjacent transpositions each cost
    one. A substring cannot be edited more than once; this differs from the
    unrestricted Damerau-Levenshtein algorithm for some longer sequences.
    """
    table = [[0] * (len(reference) + 1) for _ in range(len(prediction) + 1)]
    for row in range(len(prediction) + 1):
        table[row][0] = row
    for column in range(len(reference) + 1):
        table[0][column] = column
    for row, predicted in enumerate(prediction, 1):
        for column, gold in enumerate(reference, 1):
            table[row][column] = min(
                table[row - 1][column] + 1,
                table[row][column - 1] + 1,
                table[row - 1][column - 1] + (predicted != gold),
            )
            if (row > 1 and column > 1
                    and predicted == reference[column - 2]
                    and prediction[row - 2] == gold):
                table[row][column] = min(
                    table[row][column], table[row - 2][column - 2] + 1)
    return table[-1][-1]


def _component_tokens(pairs: list[ActionPair], components: tuple[int, ...]) -> list[Any]:
    # A partial action pair is unknown as an action token. Unknowns remain at
    # their original position, and can never equal a complete gold token.
    return [tuple(pair[index] for index in components)
            if all(pair[index] is not None for index in components) else None
            for pair in pairs]


def score_next(
    predictions: Sequence[Sequence[ActionPair]],
    references: Sequence[Sequence[ActionPair]],
    horizon: int = 3,
) -> dict[str, Any]:
    """Score a single ordered NEXT forecast at a fixed positive horizon.

    Predictions beyond ``horizon`` are ignored for accuracy and reported as a
    diagnostic. Short predictions are padded with unknown positions. References
    are likewise restricted to the horizon; short or partially labeled gold
    sequences are eligible only where the required reference component exists.
    Edit distance and exact sequence scores require a complete gold sequence
    for the scored component. No example is excluded because a prediction is
    unavailable or unmapped.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    for name, values in (("predictions", predictions), ("references", references)):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a sequence of action sequences")
    if len(predictions) != len(references):
        raise ValueError("predictions and references must contain the same number of examples")
    raw_predictions = [_pairs(sequence, f"predictions[{index}]")
                       for index, sequence in enumerate(predictions)]
    raw_references = [_pairs(sequence, f"references[{index}]")
                      for index, sequence in enumerate(references)]
    predicted = [(sequence[:horizon] + [UNKNOWN] * horizon)[:horizon]
                 for sequence in raw_predictions]
    gold = [(sequence[:horizon] + [UNKNOWN] * horizon)[:horizon]
            for sequence in raw_references]

    result: dict[str, Any] = {
        f"position_{position + 1}": score_action_pairs(
            [sequence[position] for sequence in predicted],
            [sequence[position] for sequence in gold],
        ) for position in range(horizon)
    }
    result["mean_position_accuracy"] = {}
    result["exact_ordered_sequence"] = {}
    result["normalized_edit_distance"] = {}
    result["normalized_damerau_osa"] = {}
    for name, components in COMPONENTS.items():
        rates = [result[f"position_{position + 1}"][name]["rate"]
                 for position in range(horizon)]
        eligible_rates = [rate for rate in rates if rate is not None]
        result["mean_position_accuracy"][name] = (
            sum(eligible_rates) / len(eligible_rates) if eligible_rates else None)

        eligible = [(p, g) for p, g in zip(predicted, gold)
                    if all(pair[index] is not None
                           for pair in g for index in components)]
        complete_predictions = sum(
            all(pair[index] is not None for pair in p for index in components)
            for p, _ in eligible)
        correct = sum(all(p_pair[index] is not None and p_pair[index] == g_pair[index]
                          for p_pair, g_pair in zip(p, g) for index in components)
                      for p, g in eligible)
        denominator = len(eligible)
        result["exact_ordered_sequence"][name] = {
            "correct": correct,
            "denominator": denominator,
            "rate": correct / denominator if denominator else None,
            "mapped_predictions": complete_predictions,
        }
        for metric_name, distance in (
            ("normalized_edit_distance", levenshtein_distance),
            ("normalized_damerau_osa", damerau_osa_distance),
        ):
            distance_sum = sum(distance(
                _component_tokens(p, components), _component_tokens(g, components))
                for p, g in eligible)
            result[metric_name][name] = {
                "distance_sum": distance_sum,
                "normalized_distance_sum": distance_sum / horizon,
                "denominator": denominator,
                "mean": distance_sum / (horizon * denominator) if denominator else None,
            }

    lengths = [len(sequence) for sequence in raw_predictions]
    result["forecast_diagnostics"] = {
        "examples": len(lengths),
        "horizon": horizon,
        "extracted_count": sum(lengths),
        "incomplete": sum(length < horizon for length in lengths),
        "extra": sum(length > horizon for length in lengths),
        "missing_positions": sum(max(horizon - length, 0) for length in lengths),
        "extra_actions": sum(max(length - horizon, 0) for length in lengths),
    }
    return result
