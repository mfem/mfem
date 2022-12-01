#include "mfem.hpp"
#include "gs.hpp"
#include "boundary.hpp"
#include "double_integrals.hpp"

using namespace std;
using namespace mfem;


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
      if (abs(A[j]) > tol) {
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



LinearForm * DefineRHS(PlasmaModel & model, double & rho_gamma,
                       Mesh & mesh, map<int, double> & coil_current_values,
                       ExactCoefficient & exact_coefficient,
                       ExactForcingCoefficient & exact_forcing_coeff, LinearForm & coil_term) {
  /*
    Inputs:
    model: PlasmaModel containing constants used in plasma
    attribs: unique element attributes used by the mesh
    coil_current_values: current values for each mesh attribute
    

    Outputs:
    coil_term: Linear Form of RHS

   */

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

  exact_forcing_coeff.set_coil_current(&coil_current);
  
  PWConstCoefficient coil_current_pw(coil_current);
  if (true) {
    coil_term.AddDomainIntegrator(new DomainLFIntegrator(coil_current_pw));
  }

  // manufactured solution forcing

  if (true) {
    coil_term.AddDomainIntegrator(new DomainLFIntegrator(exact_forcing_coeff));
  }

  coil_term.Assemble();

  if (true) {
    BilinearForm b(coil_term.FESpace());
    double mu = model.get_mu();
    
    auto N_lambda = [&rho_gamma, &mu](const Vector &x) -> double
    {
      return N_coefficient(x, rho_gamma, mu);
    };
    FunctionCoefficient first_boundary_coeff(N_lambda);
    b.AddBoundaryIntegrator(new MassIntegrator(first_boundary_coeff));
    auto M_lambda = [&mu](const Vector &x, const Vector &y) -> double
    {
      return M_coefficient(x, y, mu);
    };
    DoubleBoundaryBFIntegrator i(M_lambda);
    b.Assemble();
    AssembleDoubleBoundaryIntegrator(b, i, attr_ff_bdr);
    b.Finalize(); // is this needed?

    GridFunction u_ex(coil_term.FESpace());
    u_ex.ProjectCoefficient(exact_coefficient);

    b.AddMult(u_ex, coil_term);

  }
   
}

void DefineLHS(PlasmaModel & model, double rho_gamma, BilinearForm & diff_operator) {
   // Set up the bilinear form diff_operator corresponding to the diffusion integrator
   DiffusionIntegratorCoefficient diff_op_coeff(&model);
   if (true) {
     diff_operator.AddDomainIntegrator(new DiffusionIntegrator(diff_op_coeff));
   }

   // for debugging: solve I u = g
   if (false) {
     ConstantCoefficient one(1.0);
     diff_operator.AddDomainIntegrator(new MassIntegrator(one));
   }
   
   // boundary integral
   double mu = model.get_mu();
   if (true) {
     auto N_lambda = [&rho_gamma, &mu](const Vector &x) -> double
     {
       return N_coefficient(x, rho_gamma, mu);
     };

     FunctionCoefficient first_boundary_coeff(N_lambda);
     diff_operator.AddBoundaryIntegrator(new MassIntegrator(first_boundary_coeff));

     
     // BoundaryCoefficient first_boundary_coeff(rho_gamma, &model, 1);
     // diff_operator.AddBoundaryIntegrator(new MassIntegrator(first_boundary_coeff));
     // https://en.cppreference.com/w/cpp/experimental/special_functions
   }

   // assemble diff_operator
   diff_operator.Assemble();

   if (true) {
     auto M_lambda = [&mu](const Vector &x, const Vector &y) -> double
     {
       return M_coefficient(x, y, mu);
     };
     DoubleBoundaryBFIntegrator i(M_lambda);
     AssembleDoubleBoundaryIntegrator(diff_operator, i, attr_ff_bdr);
     diff_operator.Finalize(); // is this needed?
   }
   
}


void Solve(FiniteElementSpace & fespace, SysOperator & op, GridFunction & x, int & kdim, int & max_newton_iter, int & max_krylov_iter,
           double & newton_tol, double & krylov_tol) {
  
   GridFunction dx(&fespace);
   GridFunction res(&fespace);
   LinearForm out_vec(&fespace);
   dx = 0.0;

   // // SparseSmoother smoother;
   // Solver* preconditioner = new DSmoother(1);
   // GMRESSolver linear_solver;
   // linear_solver.SetKDim(kdim);
   // linear_solver.SetMaxIter(max_iter);
   // linear_solver.SetRelTol(tol);
   // linear_solver.SetAbsTol(0.0);
   // linear_solver.SetPreconditioner(*preconditioner);
   // // linear_solver.SetPreconditioner(smoother);

   // NewtonSolver newton_solver;
   // newton_solver.SetSolver(linear_solver);
   // newton_solver.SetOperator(op);
   // newton_solver.SetRelTol(tol);
   // newton_solver.SetAbsTol(0.0);
   // newton_solver.SetMaxIter(20);
   // newton_solver.SetPrintLevel(1); // print Newton iterations
   
   // Vector zero;
   // newton_solver.Mult(zero, x);
     
   double error_old;
   double error;
   LinearForm solver_error(&fespace);
   for (int i = 0; i < max_newton_iter; ++i) {

     op.Mult(x, out_vec);
     error = GetMaxError(out_vec);

     op.Mult(x, res);
     // res.Save("res.gf");

     if (i == 0) {
       printf("i: %3d, max residual: %.3e\n", i, error);
     } else {
       printf("i: %3d, max residual: %.3e, ratio %.3e\n", i, error, error_old / error);
     }
     error_old = error;

     if (error < newton_tol) {
       break;
     }

     // set<int> plasma_inds;
     // map<int, vector<int>> vertex_map;
     // vertex_map = compute_vertex_map(mesh, attr_lim);
     // int ind_min, ind_max;
     // double min_val, max_val;
     // int iprint = 0;
     // compute_plasma_points(x, mesh, vertex_map, plasma_inds, ind_min, ind_max, min_val, max_val, iprint);
     // max_val = 1.0;
     // min_val = 0.0;
     // NonlinearGridCoefficient nlgcoeff1(&model, 1, &x, min_val, max_val, plasma_inds, attr_lim);
     // GridFunction nlgc_gf(&fespace);
     // nlgc_gf.ProjectCoefficient(nlgcoeff1);
     // nlgc_gf.Save("nlgc_.gf");
     // NonlinearGridCoefficient nlgcoeff2(&model, 2, &x, min_val, max_val, plasma_inds, attr_lim);
     // GridFunction nlgc2_gf(&fespace);
     // nlgc2_gf.ProjectCoefficient(nlgcoeff2);
     // nlgc2_gf.Save("nlgc2_.gf");

     dx = 0.0;
     SparseMatrix *Mat = dynamic_cast<SparseMatrix *>(&op.GetGradient(x));

     if (i == -1) {
       // used for debugging jacobian matrix
       SparseMatrix *Compare = test_grad(&op, x, &fespace);
       // Mat->PrintMatlab();
       // Compare->PrintMatlab();
       SparseMatrix *Result;
       Result = Add(1.0, *Mat, -1.0, *Compare);
       PrintMatlab(Result, Mat, Compare);
       // Result->PrintMatlab();
     }

     GSSmoother M(*Mat);
     GMRES(*Mat, dx, out_vec, M, max_krylov_iter, kdim, krylov_tol, 0.0, 0);

     // print solver error
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
  
}


double gs(const char * mesh_file, const char * data_file, int order, int d_refine) {

   // constants associated with plasma model
   double alpha = 0.9;
   double beta = 1.5;
   double lambda = 1.0;
   double gamma = 0.9;
   double mu = 1.0;
   double r0 = 1.0;

   // boundary of far-field
   double rho_gamma = 2.5;

   map<int, double> coil_current_values;
   // 832 is the long current
   coil_current_values[832] = 0.0;
   coil_current_values[833] = 3.0;
   coil_current_values[834] = 1.0;
   coil_current_values[835] = 1.0;
   coil_current_values[836] = 1.0;
   coil_current_values[837] = 1.0;
   coil_current_values[838] = 1.0;

   // exact solution
   double r0_ = 1.0;
   double z0_ = 0.0;
   double L_ = 0.35;

   // solver options
   int kdim = 10000;
   int max_krylov_iter = 1000;
   int max_newton_iter = 5;
   double krylov_tol = 1e-12;
   double newton_tol = 1e-12;
   
   /* 
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      Process Inputs
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
   */   

   
   // Read the mesh from the given mesh file, and refine "d_refine" times uniformly.
   Mesh mesh(mesh_file);
   for (int i = 0; i < d_refine; ++i) {
     mesh.UniformRefinement();
   }
   mesh.Save("mesh.mesh");

   // save options in model
   PlasmaModel model(alpha, beta, lambda, gamma, mu, r0);

   // Define a finite element space on the mesh. Here we use H1 continuous
   // high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;
   
   double k_ = M_PI/(2.0*L_);
   ExactForcingCoefficient exact_forcing_coeff(r0_, z0_, k_, model);
   ExactCoefficient exact_coefficient(r0_, z0_, k_);

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
  DefineRHS(model, rho_gamma, mesh, coil_current_values, exact_coefficient, exact_forcing_coeff, coil_term);

   /* 
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      Define LHS
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
    */
   BilinearForm diff_operator(&fespace);
   DefineLHS(model, rho_gamma, diff_operator);

   /* 
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      Solve
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
    */
   
   // Define the solution x as a finite element grid function in fespace. Set
   // the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   GridFunction u(&fespace);
   u.ProjectCoefficient(exact_coefficient);
   u.Save("exact.gf");
   x = u;
   // now we have an initial guess: x
   // x.Save("initial_guess.gf");
   
   SysOperator op(&diff_operator, &coil_term, &model, &fespace, &mesh, attr_lim, &u);
   Solve(fespace, op, x, kdim, max_newton_iter, max_krylov_iter, newton_tol, krylov_tol);
   x.Save("final.gf");
     
   /* 
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      Error
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
    */
   GridFunction diff(&fespace);
   add(x, -1.0, u, diff);
   double num_error = GetMaxError(diff);
   diff.Save("error.gf");
   double L2_error = x.ComputeL2Error(exact_coefficient);
   printf("\n\n********************************\n");
   printf("numerical error: %.3e\n", num_error);
   printf("L2 error: %.3e\n", L2_error);
   printf("********************************\n\n");

   return L2_error;
  
}
