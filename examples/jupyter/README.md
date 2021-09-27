# Jupyter Notebooks using xeus-cling

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mfem/mfem/master?filepath=examples%2Fjupyter%2Fex.ipynb)

[xeus-cling](https://github.com/jupyter-xeus/xeus-cling) is a C++ Jupyter Kernel based on [cling](https://github.com/root-project/cling),
which can be used to create interactive C++ MFEM and GLVis notebooks.

Click on the `binder` button above for an interactive example.

## Installing Locally

In order to run notebooks locally you will need `xeus-cling` along with `mfem` and `xglvis`. We recommend you use
[miniconda](https://docs.conda.io/en/latest/miniconda.html) or, if you already have it installed,
[conda](https://docs.conda.io/projects/conda/en/latest/).

1. Follow the install steps on https://github.com/jupyter-xeus/xeus-cling to install the C++ kernels
2. Build and install a _shared_ version of mfem
    * for example: `make serial SHARED=YES`
3. Install [pyglvis](https://github.com/glvis/pyglvis)
    * for the widget frontend
4. Get [xeus-glvis](https://github.com/glvis/xeus-glvis) and `cp` the header to `{PREFIX}/glvis/xglvis.hpp`
    * (this could be improved)

## Running Locally

Once you've installed Jupyter, the C++ Kernel, mfem, and glvis start the notebook server (`jupyter-notebook`)
and open an existing example or a new `C++ 1x` kernel.

You will _always_ need to `#pragma cling load("mfem")` and you may need to point the `cling` runtime at your
mfem and/or glvis installs, do this with the
`#pragma cling` [statements](https://xeus-cling.readthedocs.io/en/latest/build_options.html#using-third-party-libraries).
