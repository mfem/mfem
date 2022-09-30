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

using namespace std;
using namespace mfem;

const int attr_r_eq_0_bdr = 831;
const int attr_ff_bdr = 900;
const int attr_lim = 1000;
const int attr_ext = 2000;


void Print_(const Vector &y) {
  for (int i = 0; i < y.Size(); ++i) {
    printf("%d %.14e\n", i+1, y[i]);
  }
}
SparseMatrix* test_grad(SysOperator *op, Vector & x, FiniteElementSpace *fespace) {

  LinearForm y1(fespace);
  LinearForm y2(fespace);

  int size = x.Size();

  double eps = 1e-4;
  SparseMatrix *Mat = new SparseMatrix(size, size);
  for (int i = 0; i < size; ++i) {
    x[i] += eps;
    op->Mult(x, y1);
    x[i] -= eps;
    op->Mult(x, y2);
    add(1.0/eps, y1, -1.0/eps, y2, y2);
    
    // Print_(y2);
    for (int j = 0; j < size; ++j) {
      Mat->Add(j, i, y2[j]);
    }
  }
  Mat->Finalize();
  // Mat->PrintMatlab();

  return Mat;
  
}

void PrintMatlab(SparseMatrix *Mat, SparseMatrix *M1, SparseMatrix *M2) {

  int *I = Mat->GetI();
  int *J = Mat->GetJ();
  double *A = Mat->GetData();
  int height = Mat->Height();

  double tol = 1e-5;
  
  int i, j;
  for (i = 0; i < height; ++i) {
    for (j = I[i]; j < I[i+1]; ++j) {
      if (A[j] > tol) {
        double m1 = 0.0;
        double m2 = 0.0;
        for (int k = M1->GetI()[i]; k < M1->GetI()[i+1]; ++k) {
          // printf("%d, %d, %d \n", k, M1->GetJ()[k], J[j]);
          if (M1->GetJ()[k] == J[j]) {
            m1 = M1->GetData()[k];
            break;
          }
        }
        for (int k = M2->GetI()[i]; k < M2->GetI()[i+1]; ++k) {
          if (M2->GetJ()[k] == J[j]) {
            m2 = M2->GetData()[k];
            break;
          }
        }
        
        
        printf("i=%d, j=%d, J=%10.3e, FD=%10.3e, diff=%10.3e \n", i, J[j], m1, m2, A[j]);
      }
    }
  }
}


int main(int argc, char *argv[])
{
   // Parse command line options.
   // const char *mesh_file = "meshes/gs_mesh.msh";
   // const char *mesh_file = "meshes/test.msh";
   const char *mesh_file = "meshes/test_off_center.msh";
   // const char *mesh_file = "meshes/square.msh";
   const char *data_file = "separated_file.data";
   int order = 1;
   int d_refine = 0;

   // constants associated with plasma model
   double alpha = 0.9;
   double beta = 1.5;
   double lambda = 1.0;
   double gamma = 0.9;
   double mu = 1.0;
   double r0 = 1.0;
   // double alpha = 5.0;
   // double beta = 1.5;
   // double lambda = 1.0;
   // double gamma = 5.0;
   // double mu = 1.0;
   // double r0 = 1.0;
   

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
   args.AddOption(&d_refine, "-g", "--refinement_factor", "Number of grid refinements");
   args.ParseCheck();

   // save options in model
   PlasmaModel model(alpha, beta, lambda, gamma, mu, r0);
   
   // unit tests
   test();
   // if (true) {
   //   return 1;
   // }
   
   // Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   for (int i = 0; i < d_refine; ++i) {
     mesh.UniformRefinement();
   }

   // Define a finite element space on the mesh. Here we use H1 continuous
   // high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // Read the data file
   // InitialCoefficient init_coeff = read_data_file(data_file);
   InitialCoefficient init_coeff = from_manufactured_solution();
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
   // ess_bdr[attr_ff_bdr-1] = 0;
   // ess_bdr[attr_ff_bdr-1] = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, boundary_dofs, 1);
   
   ConstantCoefficient one(1.0);

   /* 
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      Define RHS
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
    */
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
   if (false) {
     coil_term.AddDomainIntegrator(new DomainLFIntegrator(coil_current_pw));
   }

   // manufactured solution forcing
   double r0_ = 1.0;
   double z0_ = 0.0;
   double L_ = 0.35;
   double k_ = M_PI/(2.0*L_);
   ExactForcingCoefficient exact_forcing_coeff(r0_, z0_, k_, model);
   if (true) {
     coil_term.AddDomainIntegrator(new DomainLFIntegrator(exact_forcing_coeff));
   }

   // boundary condition
   ExactCoefficient exact_coefficient(r0_, z0_, k_);
   if (false) {
     coil_term.AddBoundaryIntegrator(new DomainLFIntegrator(exact_coefficient));
   }
   
   coil_term.Assemble();

   /* 
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      Define LHS
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
    */
   // Set up the bilinear form diff_operator corresponding to the diffusion integrator
   DiffusionIntegratorCoefficient diff_op_coeff(&model);
   BilinearForm diff_operator(&fespace);
   if (true) {
     diff_operator.AddDomainIntegrator(new DiffusionIntegrator(diff_op_coeff));
   }

   // for debugging: solve I u = g
   if (false) {
     diff_operator.AddDomainIntegrator(new MassIntegrator(one));
   }
   
   // boundary integral
   if (false) {
     BoundaryCoefficient first_boundary_coeff(rho_gamma, &model, 1);
     diff_operator.AddBoundaryIntegrator(new MassIntegrator(first_boundary_coeff));
     // https://en.cppreference.com/w/cpp/experimental/special_functions
   }

   // diff_operator.EliminateEssentialBC(boundary_dofs, Operator::DIAG_ONE);

   // assemble diff_operator
   diff_operator.Assemble();

   // Define the solution x as a finite element grid function in fespace. Set
   // the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x.ProjectCoefficient(exact_coefficient);
   x = 1.0;

   // now we have an initial guess: x
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");

   GridFunction u(&fespace);
   u.ProjectCoefficient(exact_coefficient);
   u.Save("exact.gf");

   GridFunction dx(&fespace);
   LinearForm out_vec(&fespace);
   SysOperator op(&diff_operator, &coil_term, &model, &fespace, &mesh, attr_lim, &boundary_dofs, &u);
   dx = 0.0;

   int kdim = 10000;
   int max_iter = 300;
   double tol = 1e-16;

   // SparseSmoother smoother;
   Solver* preconditioner = new DSmoother(1);
   GMRESSolver linear_solver;
   linear_solver.SetKDim(kdim);
   linear_solver.SetMaxIter(max_iter);
   linear_solver.SetRelTol(tol);
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetPreconditioner(*preconditioner);
   // linear_solver.SetPreconditioner(smoother);

   NewtonSolver newton_solver;
   newton_solver.SetSolver(linear_solver);
   newton_solver.SetOperator(op);
   newton_solver.SetRelTol(tol);
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetMaxIter(20);
   newton_solver.SetPrintLevel(1); // print Newton iterations
   
   Vector zero;
   // newton_solver.Mult(zero, x);
     
   x = u;
   double error_old;
   double error;
   LinearForm solver_error(&fespace);
   for (int i = 0; i < 20; ++i) {

     op.Mult(x, out_vec);
     error = GetMaxError(out_vec);

     GridFunction res(&fespace);
     op.Mult(x, res);
     res.Save("res.gf");
     // ConstantCoefficient zero_(0.0);
     // error = res.ComputeL2Error(zero_);

     if (i == 0) {
       // printf("\n\n********************************\n");
       printf("i: %3d, max residual: %.3e\n", i, error);
       // printf("********************************\n\n");
     } else {
       // printf("\n\n********************************\n");
       printf("i: %3d, max residual: %.3e, ratio %.3e\n", i, error, error_old / error);
       // printf("********************************\n\n");
     }
     error_old = error;

     if (error < 1e-12) {
       break;
     }

     set<int> plasma_inds;
     map<int, vector<int>> vertex_map;
     vertex_map = compute_vertex_map(mesh, attr_lim);
     int ind_min, ind_max;
     double min_val, max_val;
     int iprint = 0;
     compute_plasma_points(x, mesh, vertex_map, plasma_inds, ind_min, ind_max, min_val, max_val, iprint);
     max_val = 1.0;
     min_val = 0.0;
     NonlinearGridCoefficient nlgcoeff1(&model, 1, &x, min_val, max_val, plasma_inds, attr_lim);
     GridFunction nlgc_gf(&fespace);
     nlgc_gf.ProjectCoefficient(nlgcoeff1);
     nlgc_gf.Save("nlgc_.gf");
     NonlinearGridCoefficient nlgcoeff2(&model, 2, &x, min_val, max_val, plasma_inds, attr_lim);
     GridFunction nlgc2_gf(&fespace);
     nlgc2_gf.ProjectCoefficient(nlgcoeff2);
     nlgc2_gf.Save("nlgc2_.gf");

     dx = 0.0;
     // Operator Mat;
     // Mat = op.GetGradient(x);
     SparseMatrix *Mat = dynamic_cast<SparseMatrix *>(&op.GetGradient(x));

     if (i == -1) {
       SparseMatrix *Compare = test_grad(&op, x, &fespace);
       // Mat->PrintMatlab();
       // Compare->PrintMatlab();
       
       SparseMatrix *Result;
       Result = Add(1.0, *Mat, -1.0, *Compare);
       PrintMatlab(Result, Mat, Compare);
       // Result->PrintMatlab();
     }

     
     double tol = 1e-14;
     max_iter = 1000;
     GSSmoother M(*Mat);
     GMRES(*Mat, dx, out_vec, M, max_iter, kdim, tol, 0.0, 0);

     // Mat->Mult(dx, solver_error);
     // add(solver_error, -1.0, out_vec, solver_error);
     // double max_solver_error = GetMaxError(solver_error);
     // printf("max_solver_error: %.3e\n", max_solver_error);
       
     x -= dx;

   }
   op.Mult(x, out_vec);
   error = GetMaxError(out_vec);
   printf("\n\n********************************\n");
   printf("final max residual: %.3e, ratio %.3e\n", error, error_old / error);
   printf("********************************\n\n");

   GridFunction diff(&fespace);
   add(x, -1.0, u, diff);
   double num_error = GetMaxError(diff);
   diff.Save("error.gf");
   double L2_error = x.ComputeL2Error(exact_coefficient);
   printf("\n\n********************************\n");
   printf("numerical error: %.3e\n", num_error);
   printf("L2 error: %.3e\n", L2_error);
   printf("********************************\n\n");

   Vector x_sub;
   x.GetSubVector(boundary_dofs, x_sub);
   ofstream myfile ("dof.dat");
   x_sub.Print(myfile, 1000);   

   Vector x_exact_sub;
   u.GetSubVector(boundary_dofs, x_exact_sub);
   ofstream myfile_exact ("dof_exact.dat");
   x_exact_sub.Print(myfile_exact, 1000);   

   // Array<int> ess_vdof;
   // fespace.GetEssentialVDofs(ess_bdr, ess_vdof);
   // ofstream myfile0 ("vdof.dat"), myfile3("tdof.dat");
   // // ess_tdof_list.Print(myfile3, 1000);
   // ess_vdof.Print(myfile0, 1000);   
   
   
   x.Save("final.gf");
   return 0;
}








