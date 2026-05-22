# MTop Analysis Notebooks

This directory stores Jupyter notebooks for post-processing and visualization of `mtop` outputs.

These notebooks are:
- kept in git with the source tree
- not part of the CMake or make build
- intended to be opened directly in VS Code or Jupyter

Typical use:
1. Run an `mtop` executable to generate CSV or ParaView output.
2. Open a notebook in this directory.
3. Load data from the output directories and analyze or visualize it.

Scripts:
- `visualize_cvar_runs.py` — plot CVaR traces, evaluation counts, Armijo descents, and dual probabilities

Quick start:
```bash
cd mfem/miniapps/mtop/notebooks
python3 visualize_cvar_runs.py             # latest run
python3 visualize_cvar_runs.py --list      # list all available runs
python3 visualize_cvar_runs.py --run cvar_ # select by name prefix
```

Recommended practice:
- keep raw simulation outputs out of git
- prefer reading CSVs from relative paths near the run output
- PNG plots are saved alongside each run's CSV by default
