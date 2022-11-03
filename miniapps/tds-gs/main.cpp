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

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&data_file, "-d", "--data_file", "Plasma data file");
   args.AddOption(&d_refine, "-g", "--refinement_factor", "Number of grid refinements");
   args.AddOption(&do_test, "-t", "--test", "Perform tests only");
   args.ParseCheck();

   if (do_test == 1) {
     // unit tests
     test();
   } else {
     gs(mesh_file, data_file, order, d_refine);
   }
   
   return 0;
}








