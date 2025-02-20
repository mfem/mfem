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
making use of the L(p,q)-Jacobi family of preconditioners/smoothers.

Make sure you are familiar with the following examples:
- `ex1p` [Laplace Problem](https://github.com/mfem/mfem/blob/master/examples/ex1p.cpp)
- `ex2p` [Linear Elasticity](https://github.com/mfem/mfem/blob/master/examples/ex2p.cpp)
- `ex3p` [Definite Maxwell Problem](https://github.com/mfem/mfem/blob/master/examples/ex3p.cpp)
- `ex26p` [Multigrid Preconditioner](https://github.com/mfem/mfem/blob/master/examples/ex26p.cpp)

The code has *four* drivers: `lpq-jacobi`, `mg-lpq-jacobi`, `abs-l1-jacobi`,
and `mg-abs-l1-jacobi`. All these drivers have the capability to solve the following
problems:
- An L2-projection into a conforming H1-space.
- A diffusion problem.
- A linear elasticity problem.
- A definite Maxwell probelm.

For later reference, we say a smoother `M` is `A`-convergent if `M + M^T - A` is SPD,
this is `(Ax,x) < (Mx, x) + (M^T x, x) = 2 (Mx,x)`. It suffices to find a constant
`c < 2` such that `(Ax,x) < c(Mx,x)` to say that `M` is `A`-convergent.

# L(p,q)-Jacobi preconditioners for fully assembled systems

The driver `lpq-jacobi` allows the user solve the above mentioned problems,
utilizing a fully assembled system. The matrix of the system is required to construct
the L(p,q)-Jacobi preconditioners.

The L(p,q)-Jacobi preconditioners can be described by `D_{p,q) = diag( D^{1 + q - p}
|A|^{p} D^{-q} 1 )`, where `1` is the constant vector, `D` is the diagonal of `A`,
and the operations are understood *entrywise*. *All of this matrices are SPD*.

This code allows the uses to chose a problem (integrator) `mass, diffusion, elasticity, maxwell`,
a mesh, a solver `cg, sli`, and a Kershaw transformation to define the system. The user can
modify the frequency of the solution, the type of peconditioner `none, global, element`,
the order of the preconditioner (in case of using L(p,q)-Jacobi) `p_order, q_order`, the
polynomial degree of the underlying FES, the number of refinements (in serial and in parallel),
tolerance and maximum number of iterations for the solver, compuation on device, get `.csv`
outputs with the monitor option, and visualization to GLVis.

# Absolute-value L(1)-Jacobi preconditioner for different assembly levels

The driver `lpq-jacobi` allows the user solve the above mentioned problems,
utilizing a different types of assembly levels. Our interest lies on `AssemblyLevel::PARTIAL`.
As the FEM operator has the structure `A = P^T G^T B^T D B G P`
(see [this](https://mfem.org/performance/)), and the standard L(1)-Jacobi can be writen as
`D_1 = diag( |A|1 ). A triangle inequality implies
`D_{abs} = diag( |P^T| |G^T| |B^T| |D| |B| |G| |P| 1 )` is also `A`-convergent.

The MFEM interface allows to make use of `AbsMult` with the purpose of unwrap a composition
(of different kind) of operators as their absolute-value application. Similar run-time
options are available.

# Multigrid wrappers

The drivers `mg-lpq-jacobi` and `mg-abs-l1-jacobi` are basically the multigrid counterparts
of the previously mentioned drivers. The wrapper (akin to
[`ex26p`](https://github.com/mfem/mfem/blob/master/examples/ex26p.cpp))
allow the user to do geometric refinement or order refinement.

# Caveat: elementwise L(p,q)-Jacobi

A small interface is implemented for performing an element-based L(p,q)-Jacobi.
The function is defined in `lpq-common.xpp` and calls the element matrices,
compute the element L(p,q)-preconditioner, and then accumulates them into one operator.
