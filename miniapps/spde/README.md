# Mini-App: Surrogate Model for Imperfect Materials.


## Theory

This miniapp implements a surrogate model for imperfect materials. 
The miniapp proceeds in thee distinct steps. First, we generate a random
field $u$ by solving a fractional PDE [1] with MFEM. Second, we define a 
topological support density $v$ [2]. In the third step, we combine $u$ and $v$
to create a topology with random imperfections $w$ similar to [2]
$$ w = v +  (s \cdot T(u) + a) ,$$
where the scalar parameter `s` and `a` can be controlled via the command line 
parameters `--scale` and `--offset`, respectively. Furthermore, you may choose 
`T` as the identity transformation (`-no-urf`) or a pointwise transformation 
taking the Gaussian random field to a uniform random field (`-urf`, `-umin`, 
`-umax`).

### Fractional PDE

In the first step, we solve the fractional PDE 
$$
A^\alpha u = \eta W = f ,
$$
where $W$ is Gaussian White noise and $\eta$ a normalization constant
$$ 
\left( 
\frac{(2\pi)^{\dim{\Omega}/2} 
      \sqrt{\det{\underline{\underline{\Theta}}}}  
      \Gamma(\nu + \dim{\Omega}/2)}
     {\nu^{\dim{\Omega}/2} 
      \Gamma (\nu) } 
\right)^{1/2} .
$$
The fractional Operator A is given by
$$
A = \frac{-1}{2\nu} \nabla \circ \underline{\underline{\Theta}} \nabla + 1,
$$
the exponent $\alpha$ is defined as 
$$
\alpha = \frac{2\nu + \dim(\Omega)}{2},
$$
and the normalization $\eta$ is given by

We solve the FPDE with the same approach as in `ex33`/`ex33p`. In a nutshell, we
compute a rational approximation of the operator `A` via the triple-A algorithm.
This allows us to solve $N$ *integer-order* PDEs
$$
(A + b_i) u_i = c_i f.
$$
For more details, consider `ex33`/`ex33p` and and references [3,4,5].

The dimension is implicitly defined via the mesh, but you may specify 
$\nu$ and $\underline{\underline{\Theta}}$ via the command line arguments 
`-nu` and `-l1,-l2,-l3,-e1,-e2,-e3`.

### Topological support

For the topological support, we restrict us to functions that represent 
particles with imperfections. As defined in [2], a general function for the 
topological support is 
$$
v (x) = \tau - \rho(x,\mathcal{G}) .
$$
We follow [2, Example 2] and consider $M$ particles and define 
$$
\rho(x, \mathcal{G}) 
   = \min_{k < N } 
   \{ || \underline{\underline{S}}(x - x_k) ||_2 \} ,
$$
Where $\underline{\underline{S}}$ is a matrix indicating the shape of the
particles and $x_k$ is the position of particle $k$. Additionally, we added 
support for an octet truss. You may choose the topology with the command line 
arguments `-top` (0 = particles,1 = octet truss)

## Sample runs

Generate 5 paticles with random imperfections
```bash
mpirun -np 4 main -o 1 -r 3 -rp 3 -nu 2 \
       -l1 0.015 -l2 0.015 -l3 0.015 -s 0.01 \
       -t 0.08 -n 5 -pl2 3 -top 0 -rs
```

Generate an Octet-Truss with random imperfections
```bash
mpirun -np 4 main -o 1 -r 3 -rp 3 -nu 2 \
       -l1 0.02 -l2 0.02 -l3 0.02 -s 0.01 \
       -t 0.08 -top 1 -rs
```

Generate an Octet-Truss with random imperfections following a uniform 
distribution
```bash
mpirun -np 4 main -o 1 -r 3 -rp 3 -nu 2 \
       -l1 0.02 -l2 0.02 -l3 0.02 -umin 0.01 -umax 0.05 \
       -t 0.08 -top 1 -urf -rs
```

A 2D random field with anisotropy
```
mpirun -np 4 main -o 1 -r 3 -rp 3 -nu 4 \
       -l1 0.09 -l2 0.03 -l3 0.05 -s 0.01 \
       -t 0.08 -top 1 -no-rs -m ../../data/ref-square.mesh
```

## Visualization

The results can be visualized via GLVis or ParaView. GLVis offers quick and 
lightweight access while ParaView is a more extensive software package. 
By default, we export three scalar fields to both software packages.
* The topological support, e.g. tha basic geometrical structure
* The random perturbations
* The topology with random imperfections.
* Level set 

### GLVis

To visualize with GLVis, simply have your GLVis server running when you execute
the miniapp. Turn on/off with `-gvis/-no-gvis' command line arguments.

### ParaView

We export a file `<workdir>/ParaView/SurrogateMaterial/SurrogateMaterial.pvd` 
that can be opened and edited with ParaView as you wish. 
Turn on/off with `-pvis/-no-pvis' command line arguments. To achieve the results 
that you may see in some images of the demo, you'll have to present the 
`imperfect_topology` as volume (rendering via sampling) and use the color map 
`Xray` with separate opacity map to isolate the core structures. Note that the 
values in this miniapp and the work [2] have been chosen such that the larger 
values correspond to the topology (i.e. you want low opacity at low values and 
high opacity at high values).

## Implementation details

* The implementation is MPI parallel, if you have built MFEM without MPI, this 
  mini app will not work.
* While this mini app emphasizes [2, Example 2 and 5], users may extend this 
  mini app to other topologies (e.g. [2, Example 3] or anything else) by 
  implementing an apprpriate *distance metric*. We recommend to create a child 
  class of `MaterialTopology` (see `material_metrics.hpp`) and swap the line in 
  `synthetic_materials.cpp` in which we create the respective object.
* The matrix $\underline{\underline{\Theta}}$ can be specified with length 
  scales `-l1,-l2,-l3` and euler angles `-e1,-e2,-e3`. We construct a diagonal 
  matrix `D = diag([l1,l2,l3])` and rotation matrix `R(e1,e2,e3)` an compute 
  $\underline{\underline{\Theta}} = R^T D R$.
* The shape of the particles can be specified with `-pl1,-pl2,-pl3`. We 
  construct the matrix $\underline{\underline{S}} = R^T D R$ in the same manner
  but choose the euler angles for each particle at random.


## References:
[1] Khristenko, U., Constantinescu, A., Tallec, P. L., Oden, J. T., & 
    Wohlmuth, B. (2020). A Statistical Framework for Generating 
    Microstructures of Two-Phase Random Materials: Application to Fatigue 
    Analysis. In Multiscale Modeling &amp; Simulation (Vol. 18, Issue 1, 
    pp. 21–43). Society for Industrial & Applied Mathematics (SIAM). 
    https://doi.org/10.1137/19m1259286

[2] Khristenko, U., Constantinescu, A., Tallec, P. L., & Wohlmuth, B. (2021). 
    Statistically equivalent surrogate material models and the impact of 
    random imperfections on elasto-plastic response (Version 2). arXiv. 
    https://doi.org/10.48550/ARXIV.2112.06655

[3] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
    for rational approximation. SIAM Journal on Scientific Computing, 40(3),
    A1494-A1522.
    https://doi.org/10.1137/16M1106122

[4] Harizanov, S., Lazarov, R., Margenov, S., Marinov, P., & Pasciak, J.
    (2020). Analysis of numerical methods for spectral fractional elliptic
    equations based on the best uniform rational approximation. Journal of
    Computational Physics, 408, 109285.
    https://doi.org/10.1016/j.jcp.2020.109285

[5] Lischke, A., Pang, G., Gulian, M., Song, F., Glusa, C., Zheng, X., Mao, Z., 
    Cai, W., Meerschaert, M. M., Ainsworth, M., & Karniadakis, G. E. (2020). 
    What is the fractional Laplacian? A comparative review with new results. 
    In Journal of Computational Physics (Vol. 404, p. 109009). Elsevier BV. 
    https://doi.org/10.1016/j.jcp.2019.109009

