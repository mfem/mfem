# Miniapp: Gaussian Random Fields of Matérn Covariance for Imperfect Materials

This miniapp implements the SPDE method [1] for generating Gaussian random
fields of Matérn covariance. We use the resulting random field to model
material uncertainties similar to [2,3].

## Theory

The mini-app proceeds in three steps. First, we generate a random
field $u$ by solving a fractional PDE [1,2] with MFEM. Second, we define a
topological support density $v$ [3]. In the third step, we combine $u$ and $v$
to create a topology with random imperfections $w$ similar to [3]
$$w = v +  (s \cdot T(u) + a),$$
where the scalar parameter `s` and `a` can be controlled via the command line
parameters `--scale` and `--offset`, respectively. Furthermore, you may choose
`T` as the identity transformation (`-no-urf`) or a pointwise transformation
taking the Gaussian random field to a uniform random field (`-urf`, `-umin`,
`-umax`). The final geometry is defined as the zero level set $w=0$.

### 1. Fractional PDE

In the first step, we solve the fractional PDE
$$A^\alpha u = \eta W,$$
where $\alpha \in \mathbb{R}^+$, $W$ is Gaussian White noise,
and $\eta$ the normalization constant
$$\eta = \left( \frac{(2\pi)^{\dim{\Omega}/2} \sqrt{\det{\underline{\underline{\Theta}}}} \Gamma(\nu + \dim{\Omega}/2)} {\nu^{\dim{\Omega}/2} \Gamma (\nu) } \right)^{1/2}.$$
The fractional Operator $A$ is given by
$$A = \frac{-1}{2\nu} \nabla \circ \underline{\underline{\Theta}} \nabla + 1,$$
the exponent $\alpha$ is defined as
$$\alpha = \frac{2\nu + \dim(\Omega)}{2}.$$

We solve the FPDE with the same approach as in `ex33`/`ex33p`. In a nutshell, we
compute a rational approximation of the operator `A` via the
triple-A algorithm [4].
Instead of solving a fractional PDE, the rational approximation allows us to
solve $N$ *integer-order* PDEs
$$(A + b_i) u_i = c_i \eta W,$$
whose solutions $u_i$ approximate the true solution $u$ of the fractional
problem
$$u \approx \sum_k u_k.$$
For more details, consider `ex33`/`ex33p` and references [4,5,6].

The dimension $\dim (\Omega)$ is implicitly defined via the mesh, but you may
specify $\nu$ and $\underline{\underline{\Theta}}$ via the command line
arguments `-nu` and `-l1,-l2,-l3,-e1,-e2,-e3`. The normalization $\eta$ is
computed based on the parameter choice.

### 2. Topological support

For the topological support, we restrict ourselves to particles and an octet
truss with imperfections modeled via the random field. Following [3], a general
function for the topological support is
$$v (x) = \tau - \rho(x,\mathcal{G}).$$
and one may choose $\rho$ such that the function describes either particles or
an octet truss (c.f. [3, Example 2, Example 5]). As a user, You may specify the
topology with the command line arguments `-top` (0 = particles,1 = octet truss).

## Sample runs

Generate 5 particles with random imperfections
```bash
mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 2 \
       -l1 0.015 -l2 0.015 -l3 0.015 -s 0.01 \
       -t 0.08 -n 5 -pl2 3 -top 0 -rs
```

Generate an Octet-Truss with random imperfections
```bash
mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 2 \
       -l1 0.02 -l2 0.02 -l3 0.02 -s 0.01 \
       -t 0.08 -top 1 -rs
```

Generate an Octet-Truss with random imperfections following a uniform
distribution
```bash
mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 2 \
       -l1 0.02 -l2 0.02 -l3 0.02 -umin 0.01 -umax 0.05 \
       -t 0.08 -top 1 -urf -rs
```

A 2D random field with anisotropy
```
mpirun -np 4 generate_random_field -o 1 -r 3 -rp 3 -nu 4 \
       -l1 0.09 -l2 0.03 -l3 0.05 -s 0.01 \
       -t 0.08 -top 1 -no-rs -m ../../data/ref-square.mesh
```

## Visualization

The results can be visualized via GLVis or ParaView. GLVis offers quick and
lightweight access while ParaView is a more extensive software package.
By default, we export four scalar fields to both software packages:
* The topological support, i.e. the basic geometrical structure
* The random perturbations
* The topology with random imperfections
* The level set

### GLVis

To visualize with GLVis, simply have your GLVis server running when you execute
the mini-app. Turn on/off with `-gvis/-no-gvis' command line arguments.

### ParaView

We export a file `<workdir>/ParaView/SurrogateMaterial/SurrogateMaterial.pvd`
that can be opened and edited with ParaView as you wish.
Turn on/off with `-pvis/-no-pvis' command line arguments.

## Implementation details

* The implementation is MPI parallel, if you have built MFEM without MPI, this
  mini-app will not work.
* While the mini-app emphasizes [3, Example 2 and 5], users may extend this
  mini-app to other topologies (e.g. [2, Example 3]) by
  implementing an appropriate *distance metric*. We recommend creating a child
  class of `MaterialTopology` (see `material_metrics.hpp`) and swapping the line
  in `generate_random_field.cpp` in which we create the respective object.
* The matrix $\underline{\underline{\Theta}}$ can be specified with length
  scales `-l1,-l2,-l3` and Euler angles `-e1,-e2,-e3`. We construct a diagonal
  matrix `D = diag([l1,l2,l3])` and rotation matrix `R(e1,e2,e3)` and compute
  $\underline{\underline{\Theta}} = R^T D R$.
* The shape of the particles can be specified with `-pl1,-pl2,-pl3`. We choose
  random Euler angles for each particle.

## Accompanying presentation

[MFEM workshop 2023](https://youtu.be/s2s2YyxdTmU)

## References:

[1] Lindgren, F., Rue, H., Lindström, J. (2011). An explicit link between
    Gaussian fields and Gaussian Markov random fields: the stochastic partial
    differential equation approach. Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 73(4), 423–498.
    https://doi.org/10.1111/j.1467-9868.2011.00777.x

[2] Khristenko, U., Constantinescu, A., Tallec, P. L., Oden, J. T., &
    Wohlmuth, B. (2020). A Statistical Framework for Generating
    Microstructures of Two-Phase Random Materials: Application to Fatigue
    Analysis. In Multiscale Modeling &amp; Simulation (Vol. 18, Issue 1,
    pp. 21–43). Society for Industrial & Applied Mathematics (SIAM).
    https://doi.org/10.1137/19m1259286

[3] Khristenko, U., Constantinescu, A., Tallec, P. L., & Wohlmuth, B. (2021).
    Statistically equivalent surrogate material models and the impact of
    random imperfections on elasto-plastic response (Version 2). arXiv.
    https://doi.org/10.48550/ARXIV.2112.06655

[4] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
    for rational approximation. SIAM Journal on Scientific Computing, 40(3),
    A1494-A1522.
    https://doi.org/10.1137/16M1106122

[5] Harizanov, S., Lazarov, R., Margenov, S., Marinov, P., & Pasciak, J.
    (2020). Analysis of numerical methods for spectral fractional elliptic
    equations based on the best uniform rational approximation. Journal of
    Computational Physics, 408, 109285.
    https://doi.org/10.1016/j.jcp.2020.109285

[6] Lischke, A., Pang, G., Gulian, M., Song, F., Glusa, C., Zheng, X., Mao, Z.,
    Cai, W., Meerschaert, M. M., Ainsworth, M., & Karniadakis, G. E. (2020).
    What is the fractional Laplacian? A comparative review with new results.
    In Journal of Computational Physics (Vol. 404, p. 109009). Elsevier BV.
    https://doi.org/10.1016/j.jcp.2019.109009
