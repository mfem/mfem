"""
visualize_cvar_runs.py
-----------------------
Post-processing script for test_cvar_canti_opt_stochastic outputs.

Run from anywhere - the script searches for the cvar_csv directory upward
from this file's location.

Usage:
    python3 visualize_cvar_runs.py                     # picks latest run
    python3 visualize_cvar_runs.py --run cvar_opt_...  # selects by name prefix
    python3 visualize_cvar_runs.py --list              # lists available runs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Locate the cvar_csv directory
# ---------------------------------------------------------------------------

def find_csv_dir() -> Path:
    search = Path(__file__).resolve().parent
    first_existing = None
    for _ in range(6):
        candidates = [
            search / "build" / "miniapps" / "mtop" / "cvar_csv",
            search / "build" / "cvar_csv",
            search / "cvar_csv",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                if first_existing is None:
                    first_existing = candidate
                has_primary_csv = any(
                    p.suffix == ".csv" and not p.name.endswith("_dual_probabilities.csv")
                    for p in candidate.iterdir()
                )
                if has_primary_csv:
                    return candidate
        search = search.parent
    if first_existing is not None:
        return first_existing
    raise FileNotFoundError(
        "Could not find a cvar_csv directory in expected build locations "
        "(e.g., build/miniapps/mtop/cvar_csv) within 6 levels of this script."
    )


def list_runs(csv_dir: Path) -> list[Path]:
    return sorted(
        p for p in csv_dir.glob("*.csv")
        if not p.name.endswith("_dual_probabilities.csv")
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_main(df: pd.DataFrame, run_name: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"CVaR Optimization: {run_name}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(df["total_steps"], df["current_cvar"],           label="True CVaR (base dist.)", linewidth=2)
    ax.plot(df["total_steps"], df["current_cvar_estimation"], label="Est. CVaR (latent dist.)", linewidth=2, linestyle="--")
    ax.set_title("CVaR Traces")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Value")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["total_steps"], df["total_gradient_evaluations"], label="Gradient Evals", linewidth=2)
    ax.plot(df["total_steps"], df["total_function_evaluations"], label="Function Evals",  linewidth=2)
    ax.set_title("Evaluation Counts")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Count (cumulative)")
    ax.legend()

    ax = axes[1, 0]
    ax.bar(df["total_steps"], df["armijo_descents"], width=0.8, color="tab:red", alpha=0.75)
    ax.set_title("Armijo Descents Per Inner Iteration")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Backtracking Steps")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[1, 1]
    gap = df["current_cvar"] - df["current_cvar_estimation"]
    ax.plot(df["total_steps"], gap, color="tab:green", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("CVaR Gap  (True - Estimated)")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Difference")

    main_path = output_dir / f"{run_name}_main.png"
    plt.savefig(main_path, dpi=150)
    print(f"Saved: {main_path}")
    plt.show()

    # Log-scale views of CVaR costs.
    eps = 1e-12
    true_cvar_log = df["current_cvar"].clip(lower=eps)
    est_cvar_log = df["current_cvar_estimation"].clip(lower=eps)
    grad_evals = df["total_gradient_evaluations"]
    func_evals = df["total_function_evaluations"]
    total_evals = df["total_gradient_evaluations"] + df["total_function_evaluations"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"CVaR Costs (Log Scale): {run_name}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(df["total_steps"], true_cvar_log, linewidth=2, label="True CVaR")
    ax.plot(df["total_steps"], est_cvar_log, linewidth=2, linestyle="--", label="Estimated CVaR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("vs Total Steps")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("CVaR Cost (log scale)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(grad_evals, true_cvar_log, linewidth=2, label="True CVaR")
    ax.plot(grad_evals, est_cvar_log, linewidth=2, linestyle="--", label="Estimated CVaR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("vs Gradient Evaluations")
    ax.set_xlabel("Total Gradient Evaluations")
    ax.set_ylabel("CVaR Cost (log scale)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(func_evals, true_cvar_log, linewidth=2, label="True CVaR")
    ax.plot(func_evals, est_cvar_log, linewidth=2, linestyle="--", label="Estimated CVaR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("vs Function Evaluations")
    ax.set_xlabel("Total Function Evaluations")
    ax.set_ylabel("CVaR Cost (log scale)")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(total_evals, true_cvar_log, linewidth=2, label="True CVaR")
    ax.plot(total_evals, est_cvar_log, linewidth=2, linestyle="--", label="Estimated CVaR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("vs Total Evaluations")
    ax.set_xlabel("Total Evaluations")
    ax.set_ylabel("CVaR Cost (log scale)")
    ax.legend()

    costs_log_path = output_dir / f"{run_name}_cvar_costs_log.png"
    plt.savefig(costs_log_path, dpi=150)
    print(f"Saved: {costs_log_path}")
    plt.show()

    # True CVaR as a function of computational effort.
    total_evals = grad_evals + func_evals

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    fig.suptitle(f"CVaR vs Evaluation Counts: {run_name}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(grad_evals, df["current_cvar"], linewidth=2, marker="o", markersize=3)
    ax.set_title("True CVaR vs Gradient Evals")
    ax.set_xlabel("Total Gradient Evaluations")
    ax.set_ylabel("True CVaR")

    ax = axes[0, 1]
    ax.plot(func_evals, df["current_cvar"], linewidth=2, marker="o", markersize=3, color="tab:orange")
    ax.set_title("True CVaR vs Function Evals")
    ax.set_xlabel("Total Function Evaluations")
    ax.set_ylabel("True CVaR")

    ax = axes[0, 2]
    ax.plot(total_evals, df["current_cvar"], linewidth=2, marker="o", markersize=3, color="tab:purple")
    ax.set_title("True CVaR vs Total Evals")
    ax.set_xlabel("Total Evaluations")
    ax.set_ylabel("True CVaR")

    ax = axes[1, 0]
    ax.plot(grad_evals, df["current_cvar_estimation"], linewidth=2, marker="o", markersize=3, color="tab:green")
    ax.set_title("Estimated CVaR vs Gradient Evals")
    ax.set_xlabel("Total Gradient Evaluations")
    ax.set_ylabel("Estimated CVaR")

    ax = axes[1, 1]
    ax.plot(func_evals, df["current_cvar_estimation"], linewidth=2, marker="o", markersize=3, color="tab:red")
    ax.set_title("Estimated CVaR vs Function Evals")
    ax.set_xlabel("Total Function Evaluations")
    ax.set_ylabel("Estimated CVaR")

    ax = axes[1, 2]
    ax.plot(total_evals, df["current_cvar_estimation"], linewidth=2, marker="o", markersize=3, color="tab:brown")
    ax.set_title("Estimated CVaR vs Total Evals")
    ax.set_xlabel("Total Evaluations")
    ax.set_ylabel("Estimated CVaR")

    cvar_vs_evals_path = output_dir / f"{run_name}_cvar_vs_evals.png"
    plt.savefig(cvar_vs_evals_path, dpi=150)
    print(f"Saved: {cvar_vs_evals_path}")
    plt.show()

    # Iteration count per outer iteration with EMA trendline.
    outer_counts = (
        df.groupby("outer_iteration")["inner_iteration"]
        .count()
        .rename("iterations")
        .reset_index()
    )
    outer_counts["ema_iterations"] = outer_counts["iterations"].ewm(
        span=5,
        adjust=False,
    ).mean()

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    fig.suptitle(f"Inner Iterations Per Outer Iteration: {run_name}", fontsize=12)
    ax.plot(
        outer_counts["outer_iteration"],
        outer_counts["iterations"],
        marker="o",
        linewidth=1.8,
        label="Observed iterations",
    )
    ax.plot(
        outer_counts["outer_iteration"],
        outer_counts["ema_iterations"],
        linewidth=2.4,
        linestyle="--",
        color="tab:red",
        label="EMA (span=5)",
    )
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Number of Inner Iterations")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()

    outer_iterations_path = output_dir / f"{run_name}_outer_iterations_ema.png"
    plt.savefig(outer_iterations_path, dpi=150)
    print(f"Saved: {outer_iterations_path}")
    plt.show()


def plot_dual(dual_df: pd.DataFrame, run_name: str, output_dir: Path) -> None:
    pivot = dual_df.pivot_table(
        index="total_steps", columns="scenario_index", values="dual_probability"
    )

    # Line plot
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"Dual Probabilities: {run_name}", fontsize=12)
    for scenario_index in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[scenario_index],
            linewidth=1.5,
            alpha=0.8,
            label=f"s{scenario_index}",
        )
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Dual Probability  q_i")
    if len(pivot.columns) <= 16:
        ax.legend(ncol=2, fontsize=8)
    dual_lines_path = output_dir / f"{run_name}_dual_lines.png"
    plt.savefig(dual_lines_path, dpi=150)
    print(f"Saved: {dual_lines_path}")
    plt.show()

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"Dual Probability Heatmap: {run_name}", fontsize=12)
    im = ax.imshow(
        pivot.T.values,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[pivot.index.min(), pivot.index.max(), pivot.columns.max() + 0.5, pivot.columns.min() - 0.5],
    )
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Scenario Index")
    ax.set_yticks(pivot.columns)
    plt.colorbar(im, ax=ax, label="Dual Probability")
    dual_heatmap_path = output_dir / f"{run_name}_dual_heatmap.png"
    plt.savefig(dual_heatmap_path, dpi=150)
    print(f"Saved: {dual_heatmap_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run",  default=None, help="Name prefix to select a specific run.")
    parser.add_argument("--list", action="store_true", help="List available runs and exit.")
    args = parser.parse_args()

    csv_dir = find_csv_dir()

    runs = list_runs(csv_dir)
    if not runs:
        print(f"No CSV files found in {csv_dir}. Run the simulation first.", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print(f"Available runs in {csv_dir}:")
        for index, path in enumerate(runs):
            print(f"  [{index}] {path.stem}")
        sys.exit(0)

    if args.run:
        matches = [r for r in runs if r.stem.startswith(args.run)]
        if not matches:
            print(f"No run matching --run '{args.run}'. Use --list to see options.", file=sys.stderr)
            sys.exit(1)
        selected = matches[-1]
    else:
        selected = runs[-1]   # latest by default

    run_name = selected.stem
    print(f"Loading: {selected}")

    output_root = csv_dir / "plots"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {output_dir}")

    df = pd.read_csv(selected)
    print(df.to_string(max_rows=8))
    plot_main(df, run_name, output_dir)

    dual_path = csv_dir / f"{run_name}_dual_probabilities.csv"
    if dual_path.exists():
        print(f"Loading dual probabilities: {dual_path}")
        dual_df = pd.read_csv(dual_path)
        plot_dual(dual_df, run_name, output_dir)
    else:
        print("No dual-probability CSV found for this run; skipping dual plots.")


if __name__ == "__main__":
    main()
