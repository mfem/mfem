Solve the Grad-Shafranov equation using a newton iteration:
```
d_psi a(psi^k, v, phi^k) = l(I, v) - a(psi^k, v), for all v in V

a = + int 1/(mu r) grad psi dot grad v dr dz  (term1)
    - int (r Sp + 1/(mu r) Sff) v dr dz       (term2)
    + int_Gamma 1/mu psi(x) N(x) v(x) dS(x)   (term3)
    + int_Gamma int_Gamma 1/(2 mu) (psi(x) - psi(y)) M(x, y) (v(x) - v(y)) dS(x) dS(y)  (term4)
   
d_psi a = + int 1/(mu r) grad phi dot grad v dr dz           (term1')
          - int (r Sp' + 1/(mu r) Sff') d_psi psi_N v dr dz  (term2')
          + int_Gamma 1/mu phi(x) N(x) v(x) dS(x)            (term3')
          + int_Gamma int_Gamma 1/(2 mu) (phi(x) - phi(y)) M(x, y) (v(x) - v(y)) dS(x) dS(y)  (term4')
             
l(I, v): coil_term:     coil contribution
term1:   diff_operator: diffusion integrator
term2:   plasma_term:   nonlinear contribution from plasma
term3:   boundary term: (contained inside of diff operator)
term4:   boundary term: (contained inside of diff operator)
term1':  diff_operator:      diffusion integrator (shared with term1)
term2':  diff_plasma_term_i: derivative of nonlinear contribution from plasma (i=1,2,3)
term3':  boundary term:      (contained inside of diff operator)
term4':  boundary term:      (contained inside of diff operator)

Mesh attributes:
831:  r=0 boundary
900:  far-field boundary
1000: limiter
2000: exterior
everything else: coils
```

This code implements the algorithm described in `An adaptive Newton-based free-boundary Grad-Shafranov solver, D. A. Serino, Q. Tang, X. Tang, T. V. Kolev, and K. Lipnikov (2024)`.

Run solver for all preconditioner options, AMG cycle types, and AMG iterations for the approaches for defining p and f. 
```
run_all_2_fpol.sh: 15MA ITER baseline case
run_all_3_taylor.sh: Taylor State Equilibrium
run_all_4_lb.sh: Luxon and Brown Equilibrium
```

After running the above, the figures can be recreated with the following scripts.
```
figure 1: geometry/create_geometry.py::plot()
figure 2: process/plot_mesh.py
figure 3: process/plot_initial.py
figure 4: process/plot_iter.py
figure 5-7: process/plot_solutions.py
figure 8-9: process/plot_amr.py
```

The tables can be recreated with the following scripts.
```
table 1: see parameter selections in run_all_*.sh
table 2: process/table.py
```

Of note are the following directories
```
/data: contains data files for EFIT equilibrium used to define f and p in 15MA ITER baseline case and provide plasma control points
/figs: output folder for figures
/geometry: scripts for handling and plotting geometry
/gf: output folder for intermediate grid functions
/initial: contains initial guesses used for solver
/out_iter: output folder summarizing iteration history
/process: scripts for processing results
```

