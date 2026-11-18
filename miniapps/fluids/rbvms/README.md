# Residual-based Variational Multi-Scale

This miniapp implements the Residual-based Variational Multi-Scale (RBVMS)
method for incompressible flow.

For more details see:

[1] Y Bazilevs, VM Calo, JA Cottrell, TJR Hughes, A Reali, G.Scovazzi <br>
    [Variational multiscale residual-based turbulence modeling for large eddy simulation of incompressible flows](https://www.sciencedirect.com/science/article/pii/S0045782507003027). <br>
    CMAME Volume 197, Issues 1–4, 1 December 2007, Pages 173-201

[2] Y Bazilevs, I Akkerman <br>
    [Large eddy simulation of turbulent Taylor–Couette flow using isogeometric analysis and the residual-based variational multiscale method](https://www.sciencedirect.com/science/article/pii/S0021999110000239). <br>
    CMAME Volume 229, Issue 9, 1 May 2010, Pages 3402-3414

---
## Lid driven cavity

For a a serial computation on standard linear quadrilatoral finite elements run:

```
 rbvms -m ../../../data/inline-quad.mesh -o 1 -r 4 \
-p 0 -rho 1 -mu 0.001 \
--strong-bdr "1 2 3 4" \
-s 21 -dt 0.01 --dt_vis 0.01 -tf 10 \
-lt 1e-3 -li 250 -ni 10 -ri 10 | tee log-ldc
```

For the same simulation, but in parallel run:

```
mpirun -n 4 \
prbvms -m ../../../data/inline-quad.mesh -o 1 -r 4 \
-p 0 -rho 1 -mu 0.001 \
--strong-bdr "1 2 3 4" \
-s 21 -dt 0.01 --dt_vis 0.01 -tf 10 \
-lt 1e-3 -li 250 -ni 10 -ri 10 | tee log-ldc-par
```

We also support higher continuity NURBS simulations, for quadratic C1 discretisation run:

```
 rbvms -m ../../../data/square-nurbs.mesh -o 2 -r 4 \
-p 0 -rho 1 -mu 0.001 \
--strong-bdr 1 \
-s 21 -dt 0.01 --dt_vis 0.01 -tf 10 \
-lt 1e-3 -li 250 -ni 10 -ri 10 | tee log-ldc-nurbs
```

For the same simulation, but in parallel run:

```
mpirun -n 4 \
prbvms -m ../../../data/square-nurbs.mesh -o 2 -r 4 \
-p 0 -rho 1 -mu 0.001 \
--strong-bdr 1 \
-s 21 -dt 0.01 --dt_vis 0.01 -tf 10 \
-lt 1e-3 -li 250 -ni 10 -ri 10 | tee log-ldc-nurbs-par
```

---
## Von karman vortex street

This is a genuine unsteady benchmark. For the case of standard finite elements, 
on triangles in this case, we first need to generate a mesh.

```
gmsh von-karman-tri.geo -2 -format msh22 -clscale 0.15 
```

Note that `-clscale` controls the size of the elements generated.

For a simulation with weak enforcement of the boundary condition on the circle run:

```
 rbvms -m von-karman-tri.msh -o 1 -r 0 \
-p 1 -rho 1 -mu 0.01 \
--strong-bdr "1 2" --outflow-bdr 4  --weak-bdr 3 \
-s 45 -dt 0.1 --dt_vis 0.5 -tf 100 \
-lt 1e-3 -ni 20 -ri 10 | tee log-vkvs-tri
```

For the same simulation, but in parallel run:

```
mpirun -n 4 \
prbvms -m von-karman-tri.msh -o 1 -r 0 \
-p 1 -rho 1 -mu 0.01 \
--strong-bdr "1 2" --outflow-bdr 4  --weak-bdr 3 \
-s 45 -dt 0.1 --dt_vis 0.5 -tf 100 \
-lt 1e-3 -ni 20 -ri 10 | tee log-vkvs-tri-par
```

We also support higher continuity NURBS simulations, for quadratic C1 discretisation run:

```
 rbvms -m von-karman-nurbs.mesh -r 4 -o 2 \
-p 1 -rho 1 -mu 0.01 \
--strong-bdr "1 2" --outflow-bdr 4  --weak-bdr 3 \
-s 45 -dt 0.1 --dt_vis 0.5 -tf 100 \
-lt 1e-3 -ni 20 -ri 10 | tee log-vkvs-nurbs
```

For the same simulation, but in parallel run:

```
mpirun -n 4 \
prbvms -m von-karman-nurbs.mesh -r 4 -o 2 \
-p 1 -rho 1 -mu 0.01 \
--strong-bdr "1 2" --outflow-bdr 4  --weak-bdr 3 \
-s 45 -dt 0.1 --dt_vis 0.5 -tf 100 \
-lt 1e-3 -ni 20 -ri 10 | tee log-vkvs-nurbs-par
```

