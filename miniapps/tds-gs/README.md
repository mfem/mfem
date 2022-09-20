
./main.o -m meshes/gs_mesh.msh

After, run:
glvis -m mesh.mesh -g sol.gf

Solve the Grad-Shafranov equation using a newton iteration:
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
term3:   (contained inside of diff operator)
term4:   
term1':  diff_operator:      diffusion integrator (shared with term1)
term2':  diff_plasma_term_i: derivative of nonlinear contribution from plasma (i=1,2,3)
term3':  (contained inside of diff operator)
term4':

Mesh attributes:
831:  r=0 boundary
900:  far-field boundary
1000: limiter
2000: exterior
everything else: coils


test.cpp
test
TestCoefficient

InitialCoefficient
read_data_file

PlasmaModel
compute_plasma_points
NonlinearGridCoefficient
normalized_psi
compute_vertex_map

SysOperator
GetMaxError

DiffusionIntegratorCoefficient
one_over_r_mu


BoundaryCoefficient
DoubleIntegralCoefficient



