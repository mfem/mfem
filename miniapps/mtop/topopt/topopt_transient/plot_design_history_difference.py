#!/usr/bin/env python3
"""Plot exact Q0 control differences from paired checkpoint histories.

On the uniform Q0 control mesh used by the band-waveguide study, coefficient
Euclidean norms differ from physical L2 norms only by the common cell-volume
factor.  Optional passive-region metadata removes the fixed passive entries
from the reference norm, so the reported relative norm is over the active
control region rather than the full computational domain.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from compare_optimization_checkpoints import compare


def history_directories(root: Path):
    result = {}
    for directory in root.glob("iter_*"):
        if not directory.is_dir():
            continue
        try:
            iteration = int(directory.name.removeprefix("iter_"))
        except ValueError:
            continue
        result[iteration] = directory
    if not result:
        raise ValueError(f"{root}: no iter_XXXXXX checkpoint directories")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--do", type=Path, required=True,
                        help="DO optimization_checkpoint_history directory")
    parser.add_argument("--naive", type=Path, required=True,
                        help="naive-OD optimization_checkpoint_history directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--passive-dofs", type=int, default=0,
                        help="number of fixed passive Q0 entries")
    parser.add_argument("--passive-value", type=float, default=0.0,
                        help="common fixed value of every passive entry")
    parser.add_argument("--cell-measure", type=float, default=1.0,
                        help="uniform Q0 cell measure for physical L2 norms")
    args = parser.parse_args()
    if args.passive_dofs < 0:
        parser.error("--passive-dofs must be nonnegative")
    if args.cell_measure <= 0.0:
        parser.error("--cell-measure must be positive")

    do_history = history_directories(args.do)
    naive_history = history_directories(args.naive)
    common = sorted(set(do_history).intersection(naive_history))
    if not common:
        raise ValueError("the two checkpoint histories have no common iterations")

    rows = []
    for iteration in common:
        # Relative error is normalized by DO, the exact discrete derivative
        # route used as the declared-functional reference in this study.
        report = compare(naive_history[iteration], do_history[iteration])
        if report["metadata_compatibility_mismatches"]:
            raise ValueError(
                f"iteration {iteration}: incompatible checkpoint layouts: "
                f"{report['metadata_compatibility_mismatches']}"
            )
        passive_sq = args.passive_dofs * args.passive_value**2
        reference_active_sq = report["reference_l2_norm"]**2 - passive_sq
        candidate_active_sq = report["candidate_l2_norm"]**2 - passive_sq
        full_dot = (report["cosine"] * report["candidate_l2_norm"] *
                    report["reference_l2_norm"])
        active_dot = full_dot - passive_sq
        if reference_active_sq <= 0.0 or candidate_active_sq <= 0.0:
            raise ValueError(
                f"iteration {iteration}: passive contribution is incompatible "
                "with the checkpoint norms"
            )
        difference_norm = report["difference_l2_norm"]
        rows.append({
            "iteration": iteration,
            "relative_l2_to_do": difference_norm / reference_active_sq**0.5,
            "absolute_l2": difference_norm * args.cell_measure**0.5,
            "maximum_absolute_difference": report["maximum_absolute_error"],
            "cosine": active_dot / (candidate_active_sq *
                                      reference_active_sq)**0.5,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "design_difference_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    iterations = [row["iteration"] for row in rows]
    relative = [row["relative_l2_to_do"] for row in rows]
    maximum = [row["maximum_absolute_difference"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.3), sharex=True,
                             constrained_layout=True)
    axes[0].semilogy(iterations, relative, "o-", color="#54a24b")
    axes[0].set_ylabel(r"$\|\rho_{OD}-\rho_{DO}\|_{L^2}/\|\rho_{DO}\|_{L^2}$")
    axes[0].set_title("Paired MMA designs: exact Q0 control-space difference")
    axes[0].grid(alpha=0.25, which="both")
    axes[1].plot(iterations, maximum, "o-", color="#e45756")
    axes[1].set_xlabel("MMA iteration")
    axes[1].set_ylabel(r"$\|\rho_{OD}-\rho_{DO}\|_{L^\infty}$")
    axes[1].grid(alpha=0.25)
    fig.savefig(args.output_dir / "design_difference_history.pdf", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
