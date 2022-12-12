/* 
   Compile with: make

   Sample runs:  
   ./main.o
   ./main.o -m meshes/gs_mesh.msh
   ./main.o -m meshes/gs_mesh.msh -o 2

   After, run:
   glvis -m mesh.mesh -g sol.gf

   Description: 
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

   TODO: double boundary integral

   need boundary of plasma term?
   derivative of plasma functions?
   exact mask?
   

*/

#include "mfem.hpp"
#include <set>
#include <limits>
#include <iostream>
#include <math.h>

#include "test.hpp"
#include "exact.hpp"
#include "initial_coefficient.hpp"
#include "plasma_model.hpp"
#include "sys_operator.hpp"
#include "boundary.hpp"
#include "diffusion_term.hpp"
#include "gs.hpp"

using namespace std;
using namespace mfem;




int main(int argc, char *argv[])
{
  /* 
     -------------------------------------------------------------------------------------------
     -------------------------------------------------------------------------------------------
     -------------------------------------------------------------------------------------------
     Inputs
     -------------------------------------------------------------------------------------------
     -------------------------------------------------------------------------------------------
     -------------------------------------------------------------------------------------------
  */
  
   // Parse command line options.
   const char *mesh_file = "meshes/test_off_center.msh";
   // const char *mesh_file = "meshes/square.msh";
   const char *data_file = "separated_file.data";
   int order = 1;
   int d_refine = 0;
   int do_test = 0;

   // constants associated with plasma model
   double alpha = 0.9;
   double beta = 1.5;
   double lambda = 1.0;
   double gamma = 0.9;
   double mu = 1.0;
   double r0 = 1.0;
   // boundary of far-field
   double rho_gamma = 2.5;
   int do_manufactured_solution = 1;

   int max_krylov_iter = 1000;
   int max_newton_iter = 5;
   double krylov_tol = 1e-12;
   double newton_tol = 1e-12;

   double c1 = 0.0;
   double c2 = 3.0;
   double c3 = 1.0;
   double c4 = 1.0;
   double c5 = 1.0;
   double c6 = 1.0;
   double c7 = 1.0;
   
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&data_file, "-d", "--data_file", "Plasma data file");
   args.AddOption(&d_refine, "-g", "--refinement_factor", "Number of grid refinements");
   args.AddOption(&do_test, "-t", "--test", "Perform tests only");

   args.AddOption(&alpha, "-al", "--alpha", "alpha");
   args.AddOption(&beta, "-be", "--beta", "beta");
   args.AddOption(&lambda, "-la", "--lambda", "lambda");
   args.AddOption(&gamma, "-ga", "--gamma", "gamma");
   args.AddOption(&mu, "-mu", "--mu", "mu");
   args.AddOption(&r0, "-rz", "--r_zero", "r0");
   args.AddOption(&rho_gamma, "-rg", "--rho_gamma", "rho_gamma");
   args.AddOption(&do_manufactured_solution, "-dm", "--do_manufactured_solution", "do manufactured solution");
   args.AddOption(&max_krylov_iter, "-mk", "--max_krylov_iter", "maximum krylov iterations");
   args.AddOption(&max_newton_iter, "-mn", "--max_newton_iter", "maximum newton iterations");
   args.AddOption(&krylov_tol, "-kt", "--krylov_tol", "krylov tolerance");
   args.AddOption(&newton_tol, "-nt", "--newton_tol", "newton tolerance");
   args.AddOption(&c1, "-c1", "--c1", "coil 1");
   args.AddOption(&c2, "-c2", "--c2", "coil 2");
   args.AddOption(&c3, "-c3", "--c3", "coil 3");
   args.AddOption(&c4, "-c4", "--c4", "coil 4");
   args.AddOption(&c5, "-c5", "--c5", "coil 5");
   args.AddOption(&c6, "-c6", "--c6", "coil 6");
   args.AddOption(&c7, "-c7", "--c7", "coil 7");

   
   args.ParseCheck();

   if (do_test == 1) {
     // unit tests
     test();
   } else {
     gs(mesh_file, data_file, order, d_refine, alpha, beta, lambda, gamma, mu, r0, rho_gamma,
        max_krylov_iter, max_newton_iter, krylov_tol, newton_tol,
        c1, c2, c3, c4, c5, c6, c7,
        do_manufactured_solution);
   }
   
   return 0;
}








