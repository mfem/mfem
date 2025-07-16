```
                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|
                               https://mfem.org
```

This directory contains some drivers reimplementing basic examples in MFEM,
making use of the Abs-Value-L(1)-Jacobi family of preconditioners/smoothers.

Make sure you are familiar with the following examples:
- `ex1p` [Laplace Problem](https://github.com/mfem/mfem/blob/master/examples/ex1p.cpp)
- `ex2p` [Linear Elasticity](https://github.com/mfem/mfem/blob/master/examples/ex2p.cpp)
- `ex3p` [Definite Maxwell Problem](https://github.com/mfem/mfem/blob/master/examples/ex3p.cpp)
- `ex26p` [Multigrid Preconditioner](https://github.com/mfem/mfem/blob/master/examples/ex26p.cpp)

The code has *two* drivers: `abs-l1-jacobi` and `mg-abs-l1-jacobi`. All these
drivers have the capability to solve the following problems:
- An L2-projection into a conforming H1-space.
- A diffusion problem.
- A linear elasticity problem.
- A definite Maxwell problem.

For later reference, we say a smoother `M` is `A`-convergent if `M + M^T - A` is
SPD, this is `(Ax,x) < (Mx, x) + (M^T x, x) = 2 (Mx,x)`. It suffices to find a
constant `c < 2` such that `(Ax,x) < c(Mx,x)` to say that `M` is `A`-convergent.

# Absolute-value L(1)-Jacobi preconditioner for different assembly levels

Our interest lies on `AssemblyLevel::PARTIAL`. As the FEM operator has the
structure `A = P^T G^T B^T D B G P` (see [this](https://mfem.org/performance/)),
and the standard L(1)-Jacobi can be writen as `D_1 = diag( |A|1 ). A triangle
inequality implies `D_{abs} = diag( |P^T| |G^T| |B^T| |D| |B| |G| |P| 1 )` is
also `A`-convergent.

The MFEM interface allows to make use of `AbsMult` with the purpose of unwrap a
composition (of different kind) of operators as their absolute-value
application. Similar run-time options are available.

# Multigrid wrapper

The driver `mg-abs-l1-jacobi` is basically the multigrid counterparts of the
previously mentioned driver. The wrapper (akin to
[`ex26p`](https://github.com/mfem/mfem/blob/master/examples/ex26p.cpp))
allows the user to do geometric refinement or order refinement.
