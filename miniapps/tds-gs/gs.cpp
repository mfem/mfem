#include "mfem.hpp"
#include "gs.hpp"
#include "boundary.hpp"
#include "double_integrals.hpp"
#include <stdio.h>

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



void DefineRHS(PlasmaModelBase & model, double & rho_gamma,
               Mesh & mesh, map<int, double> & coil_current_values,
               ExactCoefficient & exact_coefficient,
               ExactForcingCoefficient & exact_forcing_coeff, LinearForm & coil_term,
               SparseMatrix * F) {
  /*
    Inputs:
    model: PlasmaModel containing constants used in plasma
    attribs: unique element attributes used by the mesh
    coil_current_values: current values for each mesh attribute
    

    Outputs:
    coil_term: Linear Form of RHS

   */
  FiniteElementSpace * fespace = coil_term.FESpace();
  int ndof = fespace->GetNDofs();
  
  int current_counter = 0;
  GridFunction ones(fespace);
  ones = 1.0;

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

      //
      Vector pw_vector(attribs.Max());
      pw_vector = 0.0;
      pw_vector(attrib-1) = 1.0;
      PWConstCoefficient pw_coeff(pw_vector);
      LinearForm lf(fespace);
      lf.AddDomainIntegrator(new DomainLFIntegrator(pw_coeff));
      lf.Assemble();
      double area = lf(ones);
      // cout << area << endl;
      for (int j = 0; j < ndof; ++j) {
        if (lf[j] != 0) {
          F->Set(j, current_counter, lf[j] / area);
        }
      }
      ++current_counter;
    }
  }
  F->Finalize();
  // F->PrintMatlab();

  exact_forcing_coeff.set_coil_current(&coil_current);

  // note: current contribution comes from F above, this is no longer necessary
  // // add contribution from currents into rhs
  // PWConstCoefficient coil_current_pw(coil_current);
  // if (false) {
  //   coil_term.AddDomainIntegrator(new DomainLFIntegrator(coil_current_pw));
  // }

  // manufactured solution forcing
  // has no effect when manufactured solution is turned off
  if (true) {
    coil_term.AddDomainIntegrator(new DomainLFIntegrator(exact_forcing_coeff));
  }

  coil_term.Assemble();

  // boundary terms
  if (true) {
    BilinearForm b(fespace);
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

    GridFunction u_ex(fespace);
    u_ex.ProjectCoefficient(exact_coefficient);

    // coil_term += b @ u_ex
    // if not a manufactured solution, we set u_ex = 0, and this term has no effect
    b.AddMult(u_ex, coil_term);

  }
   
}

void DefineLHS(PlasmaModelBase & model, double rho_gamma, BilinearForm & diff_operator) {
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


void Solve(FiniteElementSpace & fespace, SysOperator & op, GridFunction & x, int & kdim,
           int & max_newton_iter, int & max_krylov_iter,
           double & newton_tol, double & krylov_tol, double & ur_coeff,
           vector<Vector> *alpha_coeffs, vector<Array<int>> *J_inds,
           int N_control, int do_control) {
  cout << "size: " << alpha_coeffs->size() << endl;

  // todos:
  //  - weight, do control as input params
  //  - compare no control with control, but zeroing out current and multiplier
  //  - 
  
  GridFunction dx(&fespace);
  GridFunction res(&fespace);
  LinearForm out_vec(&fespace);
  dx = 0.0;

  if (do_control) {
    // solve the optimization problem of determining currents to fit desired plasma shape
    //
    // + K  dx +      + By^T dp = - opt_res
    //         + H du - F ^T dp = - reg_res
    // + By dx - F du +         = - eq_res

    // relevant matrices
    SparseMatrix * K = op.GetK();
    SparseMatrix * F = op.GetF();
    SparseMatrix * H = op.GetH();

    // solution at previous iteration
    Vector * uv = op.get_uv();
    Vector * g = op.get_g();
    GridFunction pv(&fespace);
    pv = 0.0;

    GridFunction uv_prev(&fespace);
    // uv_prev.Load("exact.gf");
    
    // placeholder for rhs
    GridFunction eq_res(&fespace);
    Vector reg_res(uv->Size());
    GridFunction opt_res(&fespace);

    // define block structure of matrix
    int n_off = 3;
    Array<int> row_offsets(n_off);
    Array<int> col_offsets(n_off);
    row_offsets[0] = 0;
    row_offsets[1] = pv.Size();
    row_offsets[2] = row_offsets[1] + pv.Size();
    
    // newton iterations
    double error_old;
    double error;
    LinearForm solver_error(&fespace);
    for (int i = 0; i < max_newton_iter; ++i) {
      // print out objective function and regularization

      // opt_res = K y^n + B_y^T p^n - g
      K->Mult(x, opt_res);
      SparseMatrix *By = dynamic_cast<SparseMatrix *>(&op.GetGradient(x));
      By->AddMultTranspose(pv, opt_res);
      add(opt_res, -1.0, *g, opt_res);

      // reg_res = H u^n - F^T p^n
      H->Mult(*uv, reg_res);
      F->AddMultTranspose(pv, reg_res, -1.0);

      // eq_res = B(y^n) - F u^n
      op.Mult(x, eq_res);

      // get max errors for residuals
      error = GetMaxError(eq_res);
      double max_opt_res = GetMaxError(opt_res);
      double max_reg_res = GetMaxError(reg_res);

      // objective + regularization
      // x^T K x - g^T x + C + uv^T H uv
      double true_obj = K->InnerProduct(x, x) - ((*g) * x) + N_control * (6.864813e-02) * (6.864813e-02);
      double regularization = (H->InnerProduct(*uv, *uv));

      // print
      if (i == 0) {
        printf("i: %3d, nonl_res: %.3e, ratio %9s, opt_res: %9s, reg_res: %9s, obj: %.3e, lst_sq: %.3e, reg: %.3e\n",
               i, error, "", "", "", true_obj+regularization, true_obj, regularization);
      } else {
        printf("i: %3d, nonl_res: %.3e, ratio %.3e, opt_res: %.3e, reg_res: %.3e, obj: %.3e, lst_sq: %.3e, reg: %.3e\n",
               i, error, error_old / error, max_opt_res, max_reg_res, true_obj+regularization,
               true_obj, regularization);
      }
      error_old = error;

      if (error < newton_tol) {
        break;
      }

      // prepare block matrix
      SparseMatrix *FT = Transpose(*F);
      SparseMatrix *mF = Add(-1.0, *F, 0.0, *F);
      SparseMatrix *mFT = Transpose(*mF);
      SparseMatrix *ByT = Transpose(*By);

      SparseMatrix *invH = new SparseMatrix(uv->Size(), uv->Size());
      for (int j = 0; j < uv->Size(); ++j) {
        invH->Set(j, j, 1.0 / (*H)(j, j));
      }
      invH->Finalize();

      SparseMatrix *invDiagByT = new SparseMatrix(pv.Size(), pv.Size());
      for (int j = 0; j < pv.Size(); ++j) {
        invDiagByT->Set(j, j, 1.0 / (*ByT)(j, j));
      }
      invDiagByT->Finalize();
      

      SparseMatrix *invHFT = Mult(*invH, *FT);
      SparseMatrix *mFinvHFT = Mult(*mF, *invHFT);
      SparseMatrix *FinvH = Mult(*F, *invH);

      // SparseMatrix *M1 = Mult(*mFinvHFT, *invDiagByT);
      SparseMatrix *M1 = Mult(*mFinvHFT, *ByT);
      SparseMatrix *M2 = Mult(*M1, *K);
      SparseMatrix *S = Add(1.0, *By, -1.0, *M2);

      // the system
      // + H du - F ^T dp         = b1 = - reg_res
      // - F du +         + By dx = b2 = - eq_res
      //        + By^T dp + K  dx = b3 = - opt_res
      //
      // is equivalent to
      //                      du* = Hinv b1*
      //          B^T dp* + K dx* = b3*
      // - F Hinv F^T dp* + B dx* = b2*
      //
      // where
      // b1* = b1             = - reg_res
      // b2* = b2 + F Hinv b1 = - eq_res - F Hinv reg_res
      // b3* = b3             = - opt_res
      //
      // dx = dx*
      // dp = dp*
      // du = du* - Hinv FT dp* = - Hinv reg_res - Hinv FT dp*
      
      // BlockMatrix *Mat;
      // Mat = new BlockMatrix(row_offsets, col_offsets);
      BlockOperator Mat(row_offsets);
      Mat.SetBlock(0, 0, ByT);
      Mat.SetBlock(0, 1, K);

      Mat.SetBlock(1, 0, mFinvHFT);
      Mat.SetBlock(1, 1, By);

      // preconditioner
      BlockDiagonalPreconditioner Prec(row_offsets);

      Solver *inv_S, *inv_ByT;
      int its = 60;
      inv_ByT = new GSSmoother(*ByT, 0, its);
      // inv_S = new GSSmoother(*S, 0, its);
      inv_S = new GSSmoother(*By, 0, its);

      Prec.SetDiagonalBlock(0, inv_ByT);
      Prec.SetDiagonalBlock(1, inv_S);

      // create block rhs vector
      BlockVector Vec(row_offsets);

      // Vec.GetBlock(0) = - opt_res;
      add(-1.0, opt_res, 0.0, opt_res, Vec.GetBlock(0));
      // Vec.GetBlock(1) = - eq_res;
      add(-1.0, eq_res, 0.0, eq_res, Vec.GetBlock(1));
      FinvH->AddMult(reg_res, Vec.GetBlock(1), -1.0);

      BlockVector dx(row_offsets);
      dx = 0.0;

      // solver
      GMRESSolver solver;
      // solver.SetAbsTol(1e-16);
      // solver.SetRelTol(1e-9);
      solver.SetAbsTol(1e-16);
      solver.SetRelTol(krylov_tol);
      solver.SetMaxIter(max_krylov_iter);
      solver.SetOperator(Mat);
      solver.SetPreconditioner(Prec);
      solver.SetKDim(kdim);
      solver.SetPrintLevel(0);

      solver.Mult(Vec, dx);
      if (solver.GetConverged())
        {
          std::cout << "GMRES converged in " << solver.GetNumIterations()
                    << " iterations with a residual norm of "
                    << solver.GetFinalNorm() << ".\n";
        }
      else
        {
          std::cout << "GMRES did not converge in " << solver.GetNumIterations()
                    << " iterations. Residual norm is " << solver.GetFinalNorm()
                    << ".\n";
        }

      x += dx.GetBlock(1);
      pv += dx.GetBlock(0);

      invH->AddMult(reg_res, *uv, -1.0);
      invHFT->AddMult(dx.GetBlock(0), *uv);

      x.Save("xtmp.gf");

    }
    // op.Mult(x, out_vec);
    // error = GetMaxError(out_vec);
    // printf("\n\n********************************\n");
    // printf("final max residual: %.3e, ratio %.3e\n", error, error_old / error);
    // printf("********************************\n\n");



     
  } else {
    // for given currents, solve the GS equations
     
    double error_old;
    double error;
    LinearForm solver_error(&fespace);
    for (int i = 0; i < max_newton_iter; ++i) {

      op.Mult(x, out_vec);
      error = GetMaxError(out_vec);
      // cout << "eq_res" << "i" << i << endl;
      // out_vec.Print();
      
      if (i == 0) {
        printf("i: %3d, max residual: %.3e\n", i, error);
      } else {
        printf("i: %3d, max residual: %.3e, ratio %.3e\n", i, error, error_old / error);
      }
      error_old = error;

      if (error < newton_tol) {
        break;
      }

      dx = 0.0;
      SparseMatrix *Mat = dynamic_cast<SparseMatrix *>(&op.GetGradient(x));

      // cout << "By" << "i" << i << endl;
      // Mat->PrintMatlab();
      
      GSSmoother M(*Mat);
      // printf("iter: %d, tol: %e, kdim: %d\n", max_krylov_iter, krylov_tol, kdim);
      int gmres_iter = max_krylov_iter;
      double gmres_tol = krylov_tol;
      int gmres_kdim = kdim;
      GMRES(*Mat, dx, out_vec, M, gmres_iter, gmres_kdim, gmres_tol, 0.0, 0);
      printf("gmres iters: %d, gmres err: %e\n", gmres_iter, gmres_tol);

      add(dx, ur_coeff - 1.0, dx, dx);
      x -= dx;

      // cout << "x" << "i" << i << endl;
      // x.Print();
      // x.Print();
      // if (true) {
      //   return;
      // }
      
    }
    op.Mult(x, out_vec);
    error = GetMaxError(out_vec);
    printf("\n\n********************************\n");
    printf("final max residual: %.3e, ratio %.3e\n", error, error_old / error);
    printf("********************************\n\n");
  }
  
}


double gs(const char * mesh_file, const char * data_file, int order, int d_refine,
          double & alpha, double & beta, double & lambda, double & gamma, double & mu,
          double & r0, double & rho_gamma, int max_krylov_iter, int max_newton_iter,
          double & krylov_tol, double & newton_tol,
          double & c1, double & c2, double & c3, double & c4, double & c5, double & c6, double & c7,
          double & c8, double & c9, double & c10, double & c11,
          double & ur_coeff,
          int do_control, double & weight,
          bool do_manufactured_solution) {

  // todo, make the below options
  // different initial condition

   map<int, double> coil_current_values;
   // center solenoids
   coil_current_values[832] = c1 / 2.71245;
   coil_current_values[833] = c2 / 2.7126;
   coil_current_values[834] = c3 / 5.4249;
   coil_current_values[835] = c4 / 2.7126;
   coil_current_values[836] = c5 / 2.71245;
   // poloidal flux coils
   coil_current_values[837] = c6 / 2.0 / 1.5;
   coil_current_values[838] = c7 / 2.0 / 1.5;
   coil_current_values[839] = c8 / 2.0 / 1.5;
   coil_current_values[840] = c9 / 2.0 / 1.5;
   coil_current_values[841] = c10 / 2.0 / 1.5;
   coil_current_values[842] = c11 / 2.0 / 1.5;

   Vector uv_currents(num_currents);
   uv_currents[0] = c1;
   uv_currents[1] = c2;
   uv_currents[2] = c3;
   uv_currents[3] = c4;
   uv_currents[4] = c5;
   uv_currents[5] = c6;
   uv_currents[6] = c7;
   uv_currents[7] = c8;
   uv_currents[8] = c9;
   uv_currents[9] = c10;
   uv_currents[10] = c11;

   // exact solution
   double r0_ = 1.0;
   double z0_ = 0.0;
   double L_ = 0.35;

   // solver options
   int kdim = 10000;

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
   // alpha: multiplier in \bar{S}_{ff'} term
   // beta: multiplier for S_{p'} term
   // gamma: multiplier for S_{ff'} term
   const char *data_file_ = "fpol_pres_ffprim_pprime.data";
   PlasmaModelFile model(mu, data_file_, alpha, beta, gamma);

   // Define a finite element space on the mesh. Here we use H1 continuous
   // high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;
   
   double k_ = M_PI/(2.0*L_);
   ExactForcingCoefficient exact_forcing_coeff(r0_, z0_, k_, model, do_manufactured_solution);
   ExactCoefficient exact_coefficient(r0_, z0_, k_, do_manufactured_solution);

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
  int ndof = fespace.GetNDofs();
  SparseMatrix * F;
  F = new SparseMatrix(ndof, num_currents);

  DefineRHS(model, rho_gamma, mesh, coil_current_values, exact_coefficient, exact_forcing_coeff, coil_term, F);
  cout << F->ActualWidth() << endl;
  cout << uv_currents.Size() << endl;
  // if (true) {
  //   return 0.0;
  // }

  SparseMatrix * H;
  H = new SparseMatrix(num_currents, num_currents);
  for (int i = 0; i < num_currents; ++i) {
    H->Set(i, i, weight);
  }
  H->Finalize();

  
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
   SparseMatrix * K;
   Vector g;
   vector<Vector> *alpha_coeffs;
   vector<Array<int>> *J_inds;
   int N_control = 10;
   if (do_manufactured_solution) {
     u.ProjectCoefficient(exact_coefficient);
     u.Save("exact.gf");
   } else {
     InitialCoefficient init_coeff = read_data_file(data_file);

     init_coeff.compute_QP(N_control, &mesh, &fespace);
     K = init_coeff.compute_K();
     g = init_coeff.compute_g();
     alpha_coeffs = init_coeff.get_alpha();
     cout << "size: " << alpha_coeffs->size() << endl;
     
     J_inds = init_coeff.get_J();
     // K->PrintMatlab();
     // g.Print();

     u.ProjectCoefficient(init_coeff);
     u.Save("initial.gf");
   }

   cout << "size: " << alpha_coeffs->size() << endl;
   x = u;
   // now we have an initial guess: x
   // x.Save("initial_guess.gf");

   SysOperator op(&diff_operator, &coil_term, &model, &fespace, &mesh, attr_lim, &u, F, &uv_currents, H, K, &g);
  // if (true) {
  //   return 0.0;
  // }
   
   Solve(fespace, op, x, kdim, max_newton_iter, max_krylov_iter, newton_tol, krylov_tol, ur_coeff,
         alpha_coeffs, J_inds, N_control, do_control);
   x.Save("final.gf");
   printf("Saved solution to final.gf\n");
   printf("Saved mesh to mesh.mesh\n");
   printf("glvis -m mesh.mesh -g final.gf\n");
     
   /* 
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      Error
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
    */

   if (do_manufactured_solution) {
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
   } else {
     return 0.0;
   }
  
}
