#!/usr/bin/env python3
"""Summarize and compare transient topology-optimization histories."""

import argparse
import itertools
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


LEGACY_COLUMNS = (
    "iteration",
    "objective",
    "volume_fraction",
    "g_upper",
    "g_lower",
    "gradient_raw_l2",
    "gradient_filtered_l2",
    "forward_seconds",
    "adjoint_seconds",
    "trajectory_mb",
    "controller_blocks",
    "local_blocks",
    "controller_intervals",
    "local_intervals",
)

CURRENT_COLUMNS = (
    "iteration",
    "objective",
    "volume_fraction",
    "g_upper",
    "g_lower",
    "gradient_raw_l2",
    "gradient_active_raw_l2",
    "gradient_filtered_l2",
    "design_change_l1",
    "forward_seconds",
    "adjoint_seconds",
    "trajectory_mb",
    "controller_blocks",
    "local_blocks",
    "controller_intervals",
    "local_intervals",
)

HEADER_ALIASES = {
    "iter": "iteration",
    "iteration": "iteration",
    "j": "objective",
    "objective": "objective",
    "vol": "volume_fraction",
    "volume": "volume_fraction",
    "vol_frac": "volume_fraction",
    "volume_fraction": "volume_fraction",
    "g": "g_upper",
    "g_upper": "g_upper",
    "g_lower": "g_lower",
    "grad_raw_l2": "gradient_raw_l2",
    "gradient_raw_l2": "gradient_raw_l2",
    "grad_filtered_l2": "gradient_filtered_l2",
    "gradient_filtered_l2": "gradient_filtered_l2",
    "grad_active_raw_l2": "gradient_active_raw_l2",
    "gradient_active_raw_l2": "gradient_active_raw_l2",
    "drho_l1": "design_change_l1",
    "design_change_l1": "design_change_l1",
    "forward_s": "forward_seconds",
    "forward_seconds": "forward_seconds",
    "adjoint_s": "adjoint_seconds",
    "adjoint_seconds": "adjoint_seconds",
    "trajectory_mb": "trajectory_mb",
    "controller_blocks": "controller_blocks",
    "local_blocks": "local_blocks",
    "controller_intervals": "controller_intervals",
    "local_intervals": "local_intervals",
}

CONFIGURATION_KEYS = (
    "problem",
    "adjoint mode",
    "time grid",
    "trajectory storage",
    "forward reconstruction",
    "discretization",
    "physics",
    "design",
    "optimizer",
    "mpi ranks",
)

COMPARISON_CONFIGURATION_KEYS = (
    "problem",
    "load",
    "band mode converter",
    "adjoint mode",
    "time grid",
    "forward reconstruction",
    "discretization",
    "physics",
    "design",
    "optimizer",
    "mpi ranks",
)


@dataclass
class HistoryRow:
    iteration: int
    values: Dict[str, float]


@dataclass
class History:
    label: str
    path: Path
    rows: Dict[int, HistoryRow] = field(default_factory=dict)
    metadata: DefaultDict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    restart_sections: int = 0
    duplicate_iterations: int = 0
    ignored_numeric_lines: int = 0


def normalized_header_name(token: str) -> str:
    key = token.strip().lower().replace("-", "_")
    return HEADER_ALIASES.get(key, key)


def header_columns(comment: str) -> Optional[Tuple[str, ...]]:
    tokens = comment.split()
    if not tokens or normalized_header_name(tokens[0]) != "iteration":
        return None
    return tuple(normalized_header_name(token) for token in tokens)


def fallback_columns(count: int) -> Optional[Tuple[str, ...]]:
    # Historical formats are prefixes of the original 14-column schema:
    # four columns used one volume constraint and five used two constraints.
    if 3 <= count <= len(LEGACY_COLUMNS):
        return LEGACY_COLUMNS[:count]
    if count == len(CURRENT_COLUMNS):
        return CURRENT_COLUMNS
    return None


def record_metadata(history: History, comment: str) -> None:
    if ":" not in comment:
        return
    key, value = comment.split(":", 1)
    key = " ".join(key.strip().lower().split())
    value = value.strip()
    if key and value and value not in history.metadata[key]:
        history.metadata[key].append(value)


def parse_history(label: str, path: Path) -> History:
    history = History(label=label, path=path)
    active_columns: Optional[Tuple[str, ...]] = None

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise ValueError("cannot read {}: {}".format(path, error)) from error

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment = stripped[1:].strip()
            # Count the section delimiter, not the following provenance lines
            # ("Restart checkpoint" and "Source checkpoint iteration").
            if comment.upper().startswith("=== RESTART"):
                history.restart_sections += 1
            parsed_header = header_columns(comment)
            if parsed_header is not None:
                active_columns = parsed_header
            else:
                record_metadata(history, comment)
            continue

        tokens = stripped.split()
        try:
            numbers = [float(token) for token in tokens]
        except ValueError:
            history.ignored_numeric_lines += 1
            print(
                "warning: {}:{} is not a numeric history row; ignored".format(
                    path, line_number
                ),
                file=sys.stderr,
            )
            continue

        columns = active_columns if active_columns and len(active_columns) == len(numbers) else None
        if columns is None:
            columns = fallback_columns(len(numbers))
        if columns is None:
            history.ignored_numeric_lines += 1
            print(
                "warning: {}:{} has unsupported {}-column schema; ignored".format(
                    path, line_number, len(numbers)
                ),
                file=sys.stderr,
            )
            continue

        iteration_value = numbers[0]
        if not math.isfinite(iteration_value) or not iteration_value.is_integer():
            history.ignored_numeric_lines += 1
            print(
                "warning: {}:{} has an invalid iteration; ignored".format(
                    path, line_number
                ),
                file=sys.stderr,
            )
            continue

        iteration = int(iteration_value)
        values = dict(zip(columns[1:], numbers[1:]))
        if iteration in history.rows:
            # A restarted run can repeat its boundary iteration.  The later row
            # is the continuation's authoritative value.
            history.duplicate_iterations += 1
        history.rows[iteration] = HistoryRow(iteration=iteration, values=values)

    if not history.rows:
        raise ValueError("{} contains no supported numeric history rows".format(path))
    return history


def finite_metric(history: History, metric: str) -> List[Tuple[int, float]]:
    result = []
    for iteration in sorted(history.rows):
        value = history.rows[iteration].values.get(metric)
        if value is not None and math.isfinite(value):
            result.append((iteration, value))
    return result


def format_number(value: float) -> str:
    return "{:.8e}".format(value)


def range_summary(history: History, metric: str, display_name: str) -> str:
    values = finite_metric(history, metric)
    if not values:
        return "  {}: unavailable".format(display_name)
    first_iteration, first_value = values[0]
    last_iteration, last_value = values[-1]
    min_iteration, min_value = min(values, key=lambda item: item[1])
    max_iteration, max_value = max(values, key=lambda item: item[1])
    return (
        "  {}: first={} (i={}), last={} (i={}), min={} (i={}), "
        "max={} (i={})"
    ).format(
        display_name,
        format_number(first_value),
        first_iteration,
        format_number(last_value),
        last_iteration,
        format_number(min_value),
        min_iteration,
        format_number(max_value),
        max_iteration,
    )


def aggregate_summary(history: History, metric: str, display_name: str) -> str:
    values = finite_metric(history, metric)
    if not values:
        return "  {}: unavailable".format(display_name)
    scalars = [value for _, value in values]
    return "  {}: total={}, mean={}, max={} (n={})".format(
        display_name,
        format_number(sum(scalars)),
        format_number(sum(scalars) / len(scalars)),
        format_number(max(scalars)),
        len(scalars),
    )


def memory_summary(history: History) -> str:
    values = finite_metric(history, "trajectory_mb")
    if not values:
        return "  trajectory memory (MB/rank): unavailable"
    scalars = [value for _, value in values]
    return "  trajectory memory (MB/rank): last={}, mean={}, peak={}".format(
        format_number(scalars[-1]),
        format_number(sum(scalars) / len(scalars)),
        format_number(max(scalars)),
    )


def nonfinite_value_count(history: History) -> int:
    return sum(
        1
        for row in history.rows.values()
        for value in row.values.values()
        if not math.isfinite(value)
    )


def print_history_summary(history: History) -> None:
    iterations = sorted(history.rows)
    print("\n== {} ==".format(history.label))
    print("file: {}".format(history.path))
    details = "{} unique iterations ({}..{})".format(
        len(iterations), iterations[0], iterations[-1]
    )
    if history.restart_sections:
        details += ", {} restart section(s)".format(history.restart_sections)
    if history.duplicate_iterations:
        details += ", {} duplicate iteration(s) replaced".format(
            history.duplicate_iterations
        )
    if history.ignored_numeric_lines:
        details += ", {} malformed row(s) ignored".format(
            history.ignored_numeric_lines
        )
    print("iterations: {}".format(details))

    configuration = []
    for key in CONFIGURATION_KEYS:
        values = history.metadata.get(key, [])
        if values:
            suffix = " (changed across sections)" if len(values) > 1 else ""
            configuration.append((key, values[-1] + suffix))
    if configuration:
        print("configuration:")
        for key, value in configuration:
            print("  {}: {}".format(key, value))

    print(range_summary(history, "objective", "objective"))
    print(range_summary(history, "volume_fraction", "volume fraction"))
    print(range_summary(history, "gradient_raw_l2", "raw gradient L2"))
    print(
        range_summary(
            history, "gradient_active_raw_l2", "active raw gradient L2"
        )
    )
    print(range_summary(history, "gradient_filtered_l2", "filtered gradient L2"))
    print(range_summary(history, "design_change_l1", "design change L1"))
    print(aggregate_summary(history, "forward_seconds", "forward time (s)"))
    print(aggregate_summary(history, "adjoint_seconds", "adjoint time (s)"))
    print(memory_summary(history))
    print(aggregate_summary(history, "controller_blocks", "controller replay blocks"))
    print(aggregate_summary(history, "local_blocks", "local replay blocks"))
    print(
        aggregate_summary(
            history, "controller_intervals", "controller replay intervals"
        )
    )
    print(aggregate_summary(history, "local_intervals", "local replay intervals"))
    nonfinite = nonfinite_value_count(history)
    if nonfinite:
        print("  non-finite numeric values: {}".format(nonfinite))
    final_fields = (
        ("final design index", "final design index"),
        ("final design objective", "final design objective"),
        ("final design volume fraction", "final design volume fraction"),
        ("final design evaluation", "final design evaluation"),
    )
    available_final = [
        (label, history.metadata[key][-1])
        for key, label in final_fields
        if history.metadata.get(key)
    ]
    if available_final:
        print("final design:")
        for label, value in available_final:
            print("  {}: {}".format(label, value))


def relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    return abs(left - right) / scale


def comparison_metric(
    left: History, right: History, common: Iterable[int], metric: str
) -> List[Tuple[int, float]]:
    differences = []
    for iteration in common:
        left_value = left.rows[iteration].values.get(metric)
        right_value = right.rows[iteration].values.get(metric)
        if (
            left_value is None
            or right_value is None
            or not math.isfinite(left_value)
            or not math.isfinite(right_value)
        ):
            continue
        differences.append(
            (iteration, relative_difference(left_value, right_value))
        )
    return differences


def print_pairwise_comparison(left: History, right: History) -> None:
    common = sorted(set(left.rows).intersection(right.rows))
    print("\n== {} vs {} ==".format(left.label, right.label))
    if not common:
        print("common iterations: none")
        return
    print(
        "common iterations: {} ({}..{}); relative denominator=max(|a|,|b|)".format(
            len(common), common[0], common[-1]
        )
    )
    metrics = (
        ("objective", "objective"),
        ("volume_fraction", "volume fraction"),
        ("gradient_raw_l2", "raw gradient L2"),
        ("gradient_active_raw_l2", "active raw gradient L2"),
        ("gradient_filtered_l2", "filtered gradient L2"),
        ("design_change_l1", "design change L1"),
    )
    for metric, display_name in metrics:
        differences = comparison_metric(left, right, common, metric)
        if not differences:
            print("  {}: unavailable on common iterations".format(display_name))
            continue
        max_iteration, max_difference = max(differences, key=lambda item: item[1])
        last_iteration, last_difference = differences[-1]
        rms = math.sqrt(
            sum(value * value for _, value in differences) / len(differences)
        )
        print(
            "  {}: max_rel={} (i={}), rms_rel={}, last_rel={} (i={}), n={}".format(
                display_name,
                format_number(max_difference),
                max_iteration,
                format_number(rms),
                format_number(last_difference),
                last_iteration,
                len(differences),
            )
        )

    for key, label in (
        ("final design objective", "final design objective"),
        ("final design volume fraction", "final design volume fraction"),
    ):
        left_values = left.metadata.get(key, [])
        right_values = right.metadata.get(key, [])
        if not left_values or not right_values:
            continue
        try:
            left_value = float(left_values[-1])
            right_value = float(right_values[-1])
        except ValueError:
            continue
        print(
            "  {}: relative difference={}".format(
                label, format_number(relative_difference(left_value, right_value))
            )
        )

    incompatible = []
    for key in COMPARISON_CONFIGURATION_KEYS:
        left_values = left.metadata.get(key, [])
        right_values = right.metadata.get(key, [])
        if left_values and right_values and left_values[-1] != right_values[-1]:
            incompatible.append(key)
    if incompatible:
        print(
            "  WARNING: non-storage configuration differs: {}".format(
                ", ".join(incompatible)
            )
        )


def parse_labeled_path(argument: str) -> Tuple[str, Path]:
    if "=" not in argument:
        raise ValueError("expected LABEL=FILE, got {!r}".format(argument))
    label, filename = argument.split("=", 1)
    label = label.strip()
    filename = filename.strip()
    if not label or not filename:
        raise ValueError("expected non-empty LABEL=FILE, got {!r}".format(argument))
    return label, Path(filename).expanduser()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize and pairwise-compare two or more transient "
            "optimization_history.txt files."
        ),
        epilog=(
            "Example: %(prog)s full=run-full/optimization_history.txt "
            "revolve=run-revolve/optimization_history.txt"
        ),
    )
    parser.add_argument(
        "histories",
        metavar="LABEL=FILE",
        nargs="+",
        help="a short run label and its history path (provide at least two)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    if len(arguments.histories) < 2:
        parser.error("provide at least two LABEL=FILE histories")

    histories = []
    labels = set()
    try:
        for argument in arguments.histories:
            label, path = parse_labeled_path(argument)
            if label in labels:
                raise ValueError("duplicate label {!r}".format(label))
            labels.add(label)
            histories.append(parse_history(label, path))
    except ValueError as error:
        parser.error(str(error))

    for history in histories:
        print_history_summary(history)
    for left, right in itertools.combinations(histories, 2):
        print_pairwise_comparison(left, right)
    return 0


if __name__ == "__main__":
    sys.exit(main())
