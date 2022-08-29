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
   TODO: initial condition
   TODO: masking
*/

#include "mfem.hpp"
#include <set>
#include <limits>
#include <iostream>
#include <math.h>

#include "test.hpp"
#include "initial_coefficient.hpp"
#include "plasma_model.hpp"
#include "sys_operator.hpp"
#include "boundary.hpp"
#include "diffusion_term.hpp"

using namespace std;
using namespace mfem;

const int attr_r_eq_0_bdr = 831;
const int attr_ff_bdr = 900;
const int attr_lim = 1000;
const int attr_ext = 2000;


int main(int argc, char *argv[])
{
   // Parse command line options.
   const char *mesh_file = "meshes/gs_mesh.msh";
   const char *data_file = "separated_file.data";
   int order = 1;

   // constants associated with plasma model
   double alpha = 2.0;
   double beta = 1.0;
   double lambda = 1.0;
   double gamma = 2.0;
   double mu = 1.0;
   double r0 = 1.0;
   // boundary of far-field
   double rho_gamma = 2.5;

   map<int, double> coil_current_values;
   // 832 is the long current
   coil_current_values[832] = 0.0;
   coil_current_values[833] = 1.0;
   coil_current_values[834] = 1.0;
   coil_current_values[835] = 1.0;
   coil_current_values[836] = 1.0;
   coil_current_values[837] = 1.0;
   coil_current_values[838] = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&mu, "-mu", "--magnetic_permeability", "Magnetic permeability of a vaccuum");
   args.AddOption(&data_file, "-d", "--data_file", "Plasma data file");
   args.ParseCheck();

   // save options in model
   PlasmaModel model(alpha, beta, lambda, gamma, mu, r0);
   
   // unit tests
   test();
   
   // Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // Define a finite element space on the mesh. Here we use H1 continuous
   // high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // Read the data file
   InitialCoefficient init_coeff = read_data_file(data_file);
   GridFunction psi_init(&fespace);
   psi_init.ProjectCoefficient(init_coeff);
   psi_init.Save("psi_init.gf");
   // if (true) {
   //   return 1;
   // }
   
   // Extract the list of all the boundary DOFs.
   // The r=0 boundary will be marked as dirichlet (psi=0)
   // and the far-field will not be marked as dirichlet
   Array<int> boundary_dofs;
   Array<int> bdr_attribs(mesh.bdr_attributes);
   Array<int> ess_bdr(bdr_attribs.Max());
   ess_bdr = 1;
   ess_bdr[attr_ff_bdr-1] = 0;
   fespace.GetEssentialTrueDofs(ess_bdr, boundary_dofs, 1);
   
   // Define the solution x as a finite element grid function in fespace. Set
   // the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // Set up the contribution from the coils
   LinearForm coil_term(&fespace);
   // these are the unique element attributes used by the mesh
   Array<int> attribs(mesh.attributes);
   Vector coil_current(attribs.Max());
   coil_current = 0.0;
   // 832 is the long coil
   for (int i = 0; i < attribs.Size(); ++i) {
     int attrib = attribs[i];
     switch(attrib) {
     case attr_ext:
       // exterior domain
       break;
     case attr_lim:
       // limiter domain
       break;
     default:
       coil_current(attrib-1) = coil_current_values[attrib];
     }
   }
   PWConstCoefficient coil_current_pw(coil_current);
   coil_term.AddDomainIntegrator(new DomainLFIntegrator(coil_current_pw));
   coil_term.Assemble();

   // Set up the bilinear form diff_operator corresponding to the diffusion integrator
   DiffusionIntegratorCoefficient diff_op_coeff(&model);
   BilinearForm diff_operator(&fespace);
   diff_operator.AddDomainIntegrator(new DiffusionIntegrator(diff_op_coeff));

   // boundary integral
   BoundaryCoefficient first_boundary_coeff(rho_gamma, &model, 1);
   diff_operator.AddBoundaryIntegrator(new MassIntegrator(first_boundary_coeff));
   // https://en.cppreference.com/w/cpp/experimental/special_functions

   // assemble diff_operator
   diff_operator.Assemble();

   // Form the linear system A X = B. This includes eliminating boundary
   // conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   diff_operator.FormLinearSystem(boundary_dofs, x, coil_term, A, X, B);

   // Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 0, 400, 1e-12, 0.0);
   diff_operator.RecoverFEMSolution(X, coil_term, x);
   // x is the recovered solution

   // now we have an initial guess: x
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");
   
   GridFunction dx(&fespace);
   LinearForm out_vec(&fespace);
   SysOperator op(&diff_operator, &coil_term, &model, &fespace, &mesh, attr_lim);
   dx = 0.0;
   for (int i = 0; i < 3; ++i) {

     op.Mult(x, out_vec);
     double error = GetMaxError(out_vec);
     printf("\n\n********************************\n");
     printf("i: %d, max error: %.3e\n", i, error);
     printf("********************************\n\n");

     int kdim = 10000;
     int max_iter = 400;
     double tol = 1e-12;
     // PCG(op.GetGradient(x), M, out_vec, dx, 0, 400, 1e-12, 0.0);
     GMRES(op.GetGradient(x), dx, out_vec, M, max_iter, kdim, tol, 0.0, 0);
     x -= dx;

   }
   op.Mult(x, out_vec);
   double error = GetMaxError(out_vec);
   printf("\n\n********************************\n");
   printf("final max error: %.3e\n", error);
   printf("********************************\n\n");
   // diff_operator.RecoverFEMSolution(X, coil_term, x);
   // double rel_tol = 1e-12;
   // // HypreSmoother *J_hypreSmoother = new HypreSmoother;
   // // J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
   // // J_hypreSmoother->SetPositiveDiagonal(true);
   // // J_prec = J_hypreSmoother;

   // MINRESSolver *J_minres = new MINRESSolver();
   // J_minres->SetRelTol(rel_tol);
   // J_minres->SetAbsTol(0.0);
   // J_minres->SetMaxIter(300);
   // J_minres->SetPrintLevel(-1);
   // // J_minres->SetPreconditioner(*J_prec);

   // NewtonSolver newton_solver;
   // newton_solver.iterative_mode = false;
   // newton_solver.SetSolver(*J_minres);
   // newton_solver.SetOperator(op);
   // newton_solver.SetPrintLevel(1); // print Newton iterations
   // newton_solver.SetRelTol(rel_tol);
   // newton_solver.SetAbsTol(0.0);
   // newton_solver.SetMaxIter(10);

   // Vector zero;
   // zero = 0.0;
   // Vector xx;
   // xx = x.GetData();
   // newton_solver.Mult(zero, xx);
   // MFEM_VERIFY(newton_solver.GetConverged(),
   //             "Newton solver did not converge.");
   

   
   
   // // now that we have solution, we can define nonlinear RHS terms
   // // plasma term
   // NonlinearGridCoefficient nlgcoeff1(&model, 1);
   // nlgcoeff1.set_grid_function(x);
   // LinearForm plasma_term(&fespace);
   // plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
   // plasma_term.Assemble();

   // // derivative of plasma term
   // NonlinearGridCoefficient nlgcoeff2(&model, 2);
   // nlgcoeff2.set_grid_function(x);
   // BilinearForm diff_plasma_term(&fespace);
   // diff_plasma_term.AddDomainIntegrator(new MassIntegrator(nlgcoeff2));
   // diff_plasma_term.Assemble();

   // // boundary term
   // if (false) {
   //   BoundaryCoefficient boundary_coeff(rho_gamma, &model, 2);
   //   DoubleIntegralCoefficient double_integral_coeff(&boundary_coeff, &fespace);
   //   double_integral_coeff.set_grid_function(&x);
   //   LinearForm boundary_term(&fespace);
   //   boundary_term.AddBoundaryIntegrator(new BoundaryLFIntegrator(double_integral_coeff));
   //   boundary_term.Assemble();
   // }
     

   //
   // GridFunction y(&fespace);
   // y= 0.0;
   // diff_operator.Mult(x, y);
   // y.Save("y.gf");

   // //
   // TestCoefficient tcoeff;
   // GridFunction z(&fespace);
   // z.ProjectCoefficient(tcoeff);
   // z.Save("z.gf");

   /*
     Test vertex to vertex mapping...
    */


   return 0;

}







