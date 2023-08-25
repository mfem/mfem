#include "mfem.hpp"
#include "gs.hpp"
#include "boundary.hpp"
#include "field.hpp"
#include "double_integrals.hpp"
#include <stdio.h>

using namespace std;
using namespace mfem;


void PrintMatlab(SparseMatrix *Mat, SparseMatrix *M1, SparseMatrix *M2);
void Print_(const Vector &y) {
  for (int i = 0; i < y.Size(); ++i) {
    printf("%d %.14e\n", i+1, y[i]);
  }
}

void test_grad(SysOperator *op, GridFunction x, FiniteElementSpace fespace) {

  LinearForm y1(&fespace);
  LinearForm y2(&fespace);
  double C1, C2, f1, f2;
  // LinearForm Cy(&fespace);
  LinearForm fy(&fespace);
  Vector * df1;
  Vector * df2;

  int size = y1.Size();

  Vector *currents = op->get_uv();
  GridFunction res_1(&fespace);
  GridFunction res_2(&fespace);
  GridFunction res_3(&fespace);
  GridFunction res_4(&fespace);
  GridFunction Cy(&fespace);
  GridFunction Ba(&fespace);
  SparseMatrix By;
  double plasma_current_1, plasma_current_2;
  double Ca;

  // *********************************
  // Test Ca and Ba
  double alpha = 1.0;
  double eps = 1e-4;

  alpha += eps;
  op->NonlinearEquationRes(x, currents, alpha);
  plasma_current_1 = op->get_plasma_current();
  res_1 = op->get_res();

  alpha -= eps;
  op->NonlinearEquationRes(x, currents, alpha);
  plasma_current_2 = op->get_plasma_current();
  Ca = op->get_Ca();

  printf("Ca: %e\n", Ca);
  
  Ba = op->get_Ba();
  res_2 = op->get_res();

  double Ca_FD = (plasma_current_1 - plasma_current_2) / eps;

  printf("\ndC/dalpha\n");
  printf("Ca: code=%e, FD=%e, Diff=%e\n", Ca, Ca_FD, Ca-Ca_FD);

  GridFunction Ba_FD(&fespace);
  add(1.0 / eps, res_1, -1.0 / eps, res_2, Ba_FD);
  printf("\ndB/dalpha\n");
  for (int i = 0; i < size; ++i) {
    if ((Ba_FD[i] != 0) || (Ba[i] != 0)) {
      printf("%d: code=%e, FD=%e, Diff=%e\n", i, Ba[i], Ba_FD[i], Ba[i]-Ba_FD[i]);
    }
  }

  // *********************************
  // Test Cy and By
  int ind_x = op->get_ind_x();
  int ind_ma = op->get_ind_ma();
  x[ind_x] += eps;
  op->NonlinearEquationRes(x, currents, alpha);
  plasma_current_1 = op->get_plasma_current();
  res_3 = op->get_res();

  x[ind_x] -= eps;
  op->NonlinearEquationRes(x, currents, alpha);
  plasma_current_2 = op->get_plasma_current();
  Cy = op->get_Cy();
  By = op->get_By();
  res_4 = op->get_res();

  double Cy_FD = (plasma_current_1 - plasma_current_2) / eps;
  
  printf("\ndC/dy\n");
  printf("Cy: code=%e, FD=%e, Diff=%e\n", Cy[ind_x], Cy_FD, Cy[ind_x]-Cy_FD);

  printf("\ndB/dy\n");
  GridFunction By_FD(&fespace);
  add(1.0 / eps, res_3, -1.0 / eps, res_4, By_FD);

  int *I = By.GetI();
  int *J = By.GetJ();
  double *A = By.GetData();
  int height = By.Height();
  for (int i = 0; i < height; ++i) {
    for (int j = I[i]; j < I[i+1]; ++j) {
      if (J[j] == ind_x) {
        printf("%d %d: code=%e, FD=%e, Diff=%e\n", i, J[j], A[j], By_FD[i], A[j]-By_FD[i]);
      }
    }
  }

  // *********************************
  // Test grad_obj and hess_obj
  double obj1, obj2, grad_obj_FD;

  op->set_i_option(2);
  
  GridFunction grad_obj(&fespace);
  GridFunction grad_obj_1(&fespace);
  GridFunction grad_obj_2(&fespace);
  GridFunction grad_obj_3(&fespace);
  GridFunction grad_obj_4(&fespace);
  grad_obj = op->compute_grad_obj(x);
  printf("\ndf/dy\n");
  for (int i = 0; i < size; ++i) {
    x[i] += eps;
    obj1 = op->compute_obj(x);
    x[i] -= eps;
    obj2 = op->compute_obj(x);

    grad_obj_FD = (obj1 - obj2) / eps;
    if ((grad_obj[i] != 0) || (grad_obj_FD != 0)) {
      printf("%d: code=%e, FD=%e, Diff=%e\n", i, grad_obj[i], grad_obj_FD, grad_obj[i]-grad_obj_FD);
    }
  }

  x[ind_x] += eps;
  grad_obj_1 = op->compute_grad_obj(x);

  x[ind_x] -= eps;
  grad_obj_2 = op->compute_grad_obj(x);

  x[ind_ma] += eps;
  grad_obj_3 = op->compute_grad_obj(x);

  x[ind_ma] -= eps;
  grad_obj_4 = op->compute_grad_obj(x);

  GridFunction K_FD_x(&fespace);
  GridFunction K_FD_ma(&fespace);
  add(1.0 / eps, grad_obj_1, -1.0 / eps, grad_obj_2, K_FD_x);
  add(1.0 / eps, grad_obj_3, -1.0 / eps, grad_obj_4, K_FD_ma);
  
  SparseMatrix * K = op->compute_hess_obj(x);

  // printf("-----\n");
  // K_FD_x.Print();
  printf("ind_x =%d\n", ind_x);
  printf("ind_ma=%d\n", ind_ma);
  
  printf("\nd2f/dy2\n");
  int *I_K = K->GetI();
  int *J_K = K->GetJ();
  double *A_K = K->GetData();
  int height_K = K->Height();
  double TOL = 1e-15;
  for (int i = 0; i < height_K; ++i) {
    for (int j = I_K[i]; j < I_K[i+1]; ++j) {
      if ((J_K[j] == ind_x) && ((abs(A_K[j]) > TOL) || (abs(K_FD_x[i]) > TOL))) {
        printf("%d %d: code=%e, FD=%e, Diff=%e\n", i, J_K[j], A_K[j], K_FD_x[i], A_K[j]-K_FD_x[i]);
      }
      if ((J_K[j] == ind_ma) && ((abs(A_K[j]) > TOL) || (abs(K_FD_ma[i]) > TOL))) {
        printf("%d %d: code=%e, FD=%e, Diff=%e\n", i, J_K[j], A_K[j], K_FD_ma[i], A_K[j]-K_FD_ma[i]);
      }
    }
  }
  
  
  
}



void PrintMatlab(SparseMatrix *Mat, SparseMatrix *M1, SparseMatrix *M2) {
  // Mat: diff
  // M1: true
  // M2: FD
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
        
        printf("i=%d, j=%d, J=%10.3e, FD=%10.3e, diff=%10.3e ", i, J[j], m1, m2, A[j] / max(m1, m2));
        if (abs(A[j] / max(m1, m2)) > 1e-4) {
          printf("***");
        }
        printf("\n");
      }
    }
  }
}



void DefineRHS(PlasmaModelBase & model, double & rho_gamma,
               Mesh & mesh, 
               ExactCoefficient & exact_coefficient,
               ExactForcingCoefficient & exact_forcing_coeff, LinearForm & coil_term,
               SparseMatrix * F) {
  /*
    8/24: This function creates the F matrix

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
    case 1100:
      break;
    default:
      double mu = model.get_mu();
      //
      Vector pw_vector(attribs.Max());
      pw_vector = 0.0;
      pw_vector(attrib-1) = 1.0;
      PWConstCoefficient pw_coeff(pw_vector);
      LinearForm lf(fespace);
      lf.AddDomainIntegrator(new DomainLFIntegrator(pw_coeff));
      lf.Assemble();
      double area = lf(ones);
      for (int j = 0; j < ndof; ++j) {
        if (lf[j] != 0) {
          F->Set(j, current_counter, lf[j] / area);
        }
      }
      ++current_counter;
    }
  }
  F->Finalize();

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

   Vector pw_vector_(2000);
   pw_vector_ = 0.0;
   pw_vector_(1100-1) = 1.0;
   PWConstCoefficient pw_coeff(pw_vector_);
   // for debugging: solve I u = g
   if (true) {
     ConstantCoefficient one(1.0);
     diff_operator.AddDomainIntegrator(new MassIntegrator(pw_coeff));
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


void Solve(FiniteElementSpace & fespace, PlasmaModelBase *model, GridFunction & x, int & kdim,
           int & max_newton_iter, int & max_krylov_iter,
           double & newton_tol, double & krylov_tol, 
           double & Ip, int N_control, int do_control,
           int add_alpha, int obj_option, double & obj_weight,
           double & rho_gamma,
           Mesh * mesh,
           ExactForcingCoefficient * exact_forcing_coeff,
           ExactCoefficient * exact_coefficient,
           InitialCoefficient * init_coeff,
           bool include_plasma,
           double & weight_coils,
           double & weight_solenoids,
           Vector * uv,
           double & alpha) {

   // 9. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   bool visualization = true;
   // if (visualization)
   // {
   //    sol_sock.open(vishost, visport);
   // }
  
  // don't need to recompute matrices everytime
  // save mesh after refining
  
  GridFunction xnew(&fespace);
  GridFunction res(&fespace);

  GridFunction dx_(&fespace);
  GridFunction dp_(&fespace);
  double da_;
  double dl_;

  SparseMatrix *invH;
  SparseMatrix *invHFT;

  // bool add_alpha = true;
  bool reduce = true;

  GridFunction psi_r(&fespace);
  GridFunction psi_z(&fespace);

  FieldCoefficient BrCoeff(&x, &psi_r, &psi_z, model, fespace, 0);
  FieldCoefficient BpCoeff(&x, &psi_r, &psi_z, model, fespace, 1);
  FieldCoefficient BzCoeff(&x, &psi_r, &psi_z, model, fespace, 2);
  GridFunction Br_field(&fespace);
  GridFunction Bp_field(&fespace);
  GridFunction Bz_field(&fespace);

  // Save data in the VisIt format
  VisItDataCollection visit_dc("gs", fespace.GetMesh());
  visit_dc.RegisterField("psi", &x);
  visit_dc.RegisterField("Br", &Br_field);
  visit_dc.RegisterField("Bp", &Bp_field);
  visit_dc.RegisterField("Bz", &Bz_field);

  int max_levels = 2;
  int max_dofs = 12000;

  if (do_control) {
    // solve the optimization problem of determining currents to fit desired plasma shape
    // + K  dx +      + By^T dp = - opt_res
    //         + H du - F ^T dp = - reg_res
    // + By dx - F du +         = - eq_res
    //
    // 0-beta problem
    // + K   dx +      + By^T dp +       + Cy dl = b1 = - opt_res
    //          + H du - F ^T dp +       +       = b2 = - reg_res
    // + By  dx - F du +         + Ba da +       = b3 = - eq_res
    // +        +      + Ba^T dp +       + Ca dl = b4
    // + CyT dx +      +         + Ca da +       = b5

    // solution at previous iteration
    printf("currents: [");
    for (int i = 0; i < uv->Size(); ++i) {
      printf("%.3e ", (*uv)[i]);
    }
    printf("]\n");

    GridFunction pv(&fespace);
    pv = 0.0;
    double lv = 0.0;

    // placeholder for rhs
    GridFunction eq_res(&fespace);
    Vector reg_res(uv->Size());
    GridFunction opt_res(&fespace);
    opt_res = 0.0;

    GridFunction b1(&fespace);
    Vector b2(uv->Size());
    GridFunction b3(&fespace);
    b1 = 0.0;
    b2 = 0.0;
    b3 = 0.0;

    GridFunction b1p(&fespace);
    Vector b2p(uv->Size());
    GridFunction b3p(&fespace);
    b1p = 0.0;
    b2p = 0.0;
    b3p = 0.0;

    // define block structure of matrix
    int n_off = 6;
    if (reduce) {
      n_off = 5;
    }
    Array<int> row_offsets(n_off);
    Array<int> col_offsets(n_off);

    // newton iterations
    double error_old;
    double error;
    for (int i = 0; i < max_newton_iter; ++i) {

      xnew = x;

      // define error estimator for AMR
      ErrorEstimator *estimator{nullptr};
      ConstantCoefficient one(1.0);
      // BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
      DiffusionIntegratorCoefficient diff_op_coeff(model);
      DiffusionIntegrator *integ = new DiffusionIntegrator(diff_op_coeff);
      estimator = new LSZienkiewiczZhuEstimator(*integ, xnew);
      ThresholdRefiner refiner(*estimator);
      refiner.SetTotalErrorFraction(0.7);
      
      for (int it_amr = 0; ; ++it_amr) {
        int cdofs = fespace.GetTrueVSize();
        cout << "\nAMR iteration " << it_amr << endl;
        cout << "Number of unknowns: " << cdofs << endl;

        
        LinearForm coil_term(&fespace);
        int ndof__ = fespace.GetNDofs();
        SparseMatrix * F;
        F = new SparseMatrix(ndof__, num_currents);
        DefineRHS(*model, rho_gamma, *mesh, *exact_coefficient, *exact_forcing_coeff, coil_term, F);
  
        BilinearForm diff_operator(&fespace);
        DefineLHS(*model, rho_gamma, diff_operator);

        SparseMatrix * K_;
        Vector g_;
        vector<Vector> *alpha_coeffs;
        vector<Array<int>> *J_inds;
        alpha_coeffs = new vector<Vector>;
        J_inds = new vector<Array<int>>;
        init_coeff->compute_QP(N_control, mesh, &fespace);
        K_ = init_coeff->compute_K();
        g_ = init_coeff->compute_g();
        J_inds = init_coeff->get_J();
        alpha_coeffs = init_coeff->get_alpha();

        SparseMatrix * H;
        H = new SparseMatrix(num_currents, num_currents);
        for (int i = 0; i < num_currents; ++i) {
          if (i < 5) {
            H->Set(i, i, weight_coils);
          } else {
            H->Set(i, i, weight_solenoids);
          }
        }
        H->Finalize();

        SysOperator op(&diff_operator, &coil_term, model, &fespace, mesh, attr_lim, &x, F, uv, H, K_, &g_, alpha_coeffs, J_inds, &alpha, include_plasma);
        op.set_i_option(obj_option);
        op.set_obj_weight(obj_weight);  
        
        if (reduce) {
          row_offsets[0] = 0;
          row_offsets[1] = pv.Size();
          row_offsets[2] = row_offsets[1] + pv.Size();
          row_offsets[3] = row_offsets[2] + 1;
          row_offsets[4] = row_offsets[3] + 1;
        } else {
          row_offsets[0] = 0;
          row_offsets[1] = pv.Size();
          row_offsets[2] = row_offsets[1] + uv->Size();
          row_offsets[3] = row_offsets[2] + pv.Size();
          row_offsets[4] = row_offsets[3] + 1;
          row_offsets[5] = row_offsets[4] + 1;
        }

        // print out objective function and regularization

        // -b3 = eq_res = B(y^n) - F u^n
        op.NonlinearEquationRes(x, uv, alpha);
        eq_res = op.get_res();
        SparseMatrix By = op.get_By();
        double C = op.get_plasma_current();
        double Ca = op.get_Ca();
        Vector Cy = op.get_Cy();
        Vector Ba = op.get_Ba();
        cout << "plasma current " << C / op.get_mu() << endl;

        // if (i == 0) {
        //   eq_res.Save("gf/initial_eq_res.gf");
        // } else {
        //   eq_res.Save("gf/eq_res.gf");
        // }
        char name_eq_res[60];
        sprintf(name_eq_res, "gf/eq_res_i%d_amr%d.gf", i, it_amr);
        eq_res.Save(name_eq_res);

        char name_mesh[60];
        sprintf(name_mesh, "gf/mesh_i%d_amr%d.mesh", i, it_amr);
        mesh->Save(name_mesh);

        // if (visualization && sol_sock.good())
        //   {
        //     sol_sock.precision(8);
        //     sol_sock << "solution\n" << (*mesh) << x << flush;
        //   }
        double psi_x = op.get_psi_x();
        double psi_ma = op.get_psi_ma();
        double* x_x = op.get_x_x();
        double* x_ma = op.get_x_ma();

        printf("psi_x = %e; r_x = %e; z_x = %e\n", psi_x, x_x[0], x_x[1]);
        printf("psi_ma = %e; r_ma = %e; z_ma = %e\n", psi_ma, x_ma[0], x_ma[1]);

        // if (i == 0) {
        //   return;
        // }
        BrCoeff.set_psi_vals(psi_x, psi_ma);
        BpCoeff.set_psi_vals(psi_x, psi_ma);
        BzCoeff.set_psi_vals(psi_x, psi_ma);

        SparseMatrix * K = op.compute_hess_obj(x);

        printf("K: %d %d\n", K->Height(), K->ActualWidth());
        printf("xv size: %d\n", x.Size());
      
        Vector g = op.compute_grad_obj(x);

        opt_res = g;
        By.AddMultTranspose(pv, opt_res);
        if (add_alpha) {
          add(opt_res, lv, Cy, opt_res);
        }

        // -b2 = reg_res = H u^n - F^T p^n
        H->Mult(*uv, reg_res);
        F->AddMultTranspose(pv, reg_res, -1.0);

        if (add_alpha) {
          cout << "alpha: " << alpha << endl;
        }

        // b4 = B_a^T p^n + C_a l^n
        double b4 = Ba * pv + Ca * lv;
        b4 *= -1.0;

        double b5= C - Ip * op.get_mu();
        b5 *= -1.0;

        // get max errors for residuals
        error = GetMaxError(eq_res);
        double max_opt_res = op.get_mu() * GetMaxError(opt_res);
        double max_reg_res = GetMaxError(reg_res) / op.get_mu();

        // objective + regularization
        // x^T K x - g^T x + C + uv^T H uv
        double true_obj = K->InnerProduct(x, x);
        true_obj *= 0.5;
        double test_obj = op.compute_obj(x);
        printf("objective test: yKy=%e, formula=%e, diff=%e\n", true_obj, test_obj, true_obj - test_obj);
      
        double regularization = (H->InnerProduct(*uv, *uv));

        if (i == 0) {
          printf("i: %3d, nonl_res: %.3e, ratio %9s, res: [%.3e, %.3e, %.3e, %.3e], loss: %.3e, obj: %.3e, reg: %.3e\n",
                 i, error, "", max_opt_res, max_reg_res, abs(b4), abs(b5), true_obj+regularization, test_obj, regularization);
        } else {
          printf("i: %3d, nonl_res: %.3e, ratio %.3e, res: [%.3e, %.3e, %.3e, %.3e], loss: %.3e, obj: %.3e, reg: %.3e\n",
                 i, error, error_old / error, max_opt_res, max_reg_res, abs(b4), abs(b5), test_obj+regularization,
                 test_obj, regularization);
        }
        error_old = error;
        printf("\n");

        if (error < newton_tol) {
          break;
        }

        // prepare block matrix
        SparseMatrix *FT = Transpose(*F);
        SparseMatrix *mF = Add(-1.0, *F, 0.0, *F);
        SparseMatrix *mFT = Transpose(*mF);
        SparseMatrix *ByT = Transpose(By);

        invH = new SparseMatrix(uv->Size(), uv->Size());
        for (int j = 0; j < uv->Size(); ++j) {
          invH->Set(j, j, 1.0 / (*H)(j, j));
        }
        invH->Finalize();

        invHFT = Mult(*invH, *FT);
        SparseMatrix *mFinvHFT = Mult(*mF, *invHFT);
        SparseMatrix *FinvH = Mult(*F, *invH);
        SparseMatrix *mMuFinvHFT = Add(op.get_mu(), *mFinvHFT, 0.0, *mFinvHFT);
        SparseMatrix *MuFinvH = Add(op.get_mu(), *FinvH, 0.0, *FinvH);

        SparseMatrix *Ba_;
        SparseMatrix *Cy_;
        SparseMatrix *Ca_;
        SparseMatrix *inv_Ca_;
        Ba_ = new SparseMatrix(pv.Size(), 1);
        Cy_ = new SparseMatrix(pv.Size(), 1);
        Ca_ = new SparseMatrix(1, 1);
        inv_Ca_ = new SparseMatrix(1, 1);
        Ca_->Add(0, 0, Ca);
        inv_Ca_->Add(0, 0, 1.0 / Ca);
        for (int j = 0; j < pv.Size(); ++j) {
          if (Ba[j] != 0) {
            Ba_->Add(j, 0, Ba[j]);
          }
          if (Cy[j] != 0) {
            Cy_->Add(j, 0, Cy[j]);
          }
        }
        Ca_->Finalize();
        inv_Ca_->Finalize();
        Ba_->Finalize();
        Cy_->Finalize();
        SparseMatrix *CyT_ = Transpose(*Cy_);
        SparseMatrix *BaT_ = Transpose(*Ba_);

        // *******************************************************************
        // the system
        //        + (ByT - 1 / Ca Cy BaT) dp + K dx = b1p = b1 - 1 / Ca Cy b4
        // + H du - F ^T dp                         = b2p = b2
        // - F du + (By  - 1 / Ca Ba CyT) dx        = b3p = b3 - 1 / Ca Ba b5
        //
        // is equivalent to
        //                              B^T dp* + (K - 1 / Ca Cy CyT) dx* = b1*
        //                                                            du* = Hinv b2*
        // (- 1 / Ca Ba Ba^T - F Hinv F^T ) dp* +                   B dx* = b3*
        //
        // where
        // b1* = b1p
        // b2* = b2p
        // b3* = b3p + F Hinv b2p
        // 
        // dx = dx*
        // dp = dp*
        // du = du* - Hinv FT dp*

        // By = Add(op.get_mu(), *By, 0.0, *By);
        // Ba_ = Add(op.get_mu(), *Ba_, 0.0, *Ba_);
        // // das: preconditioner debug
        // // need to go into equations, can't just mult by mu
      
        // get b1, b2, b3
        b1 = opt_res; b1 *= -1.0;
        b2 = reg_res; b2 *= -1.0;
        b3 = eq_res;  b3 *= -1.0;
      
        BlockOperator Mat(row_offsets);
        Mat.SetBlock(0, 0, &By);
        Mat.SetBlock(0, 1, mMuFinvHFT);
        if (add_alpha) {
          Mat.SetBlock(0, 2, Ba_);
        }

        Mat.SetBlock(1, 0, K);
        Mat.SetBlock(1, 1, ByT);
        if (add_alpha) {
          Mat.SetBlock(1, 3, Cy_);
        }

        Mat.SetBlock(2, 2, Ca_);
        if (add_alpha) {
          Mat.SetBlock(2, 0, CyT_);
        }
        
        Mat.SetBlock(3, 3, Ca_);
        if (add_alpha) {
          Mat.SetBlock(3, 1, BaT_);
        }
      
        // *******************************************************************
        // preconditioner
        BlockDiagonalPreconditioner Prec(row_offsets);

        Solver *inv_ByT, *inv_By;
        int its = 60;
        inv_ByT = new GSSmoother(*ByT, 0, its);
        inv_By = new GSSmoother(By, 0, its);

        Prec.SetDiagonalBlock(0, inv_By);
        Prec.SetDiagonalBlock(1, inv_ByT);
        Prec.SetDiagonalBlock(2, inv_Ca_);
        Prec.SetDiagonalBlock(3, inv_Ca_);

        // *******************************************************************
        // create block rhs vector
        BlockVector Vec(row_offsets);

        Vec.GetBlock(0) = b3;
        MuFinvH->AddMult(b2, Vec.GetBlock(0));
        Vec.GetBlock(1) = b1;
        Vec.GetBlock(2) = b5;
        Vec.GetBlock(3) = b4;

        // *******************************************************************
        // solver
        GMRESSolver solver;
        solver.SetAbsTol(1e-16);
        solver.SetRelTol(krylov_tol);
        solver.SetMaxIter(max_krylov_iter);
        solver.SetOperator(Mat);
        solver.SetPreconditioner(Prec);
        solver.SetKDim(kdim);
        solver.SetPrintLevel(0);

        BlockVector dx(row_offsets);
        dx = 0.0;
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

        dx_ = dx.GetBlock(0);
        dp_ = dx.GetBlock(1);
        da_ = dx.GetBlock(2)[0];
        dl_ = dx.GetBlock(3)[0];
        add(x, dx_, xnew);

        if (it_amr >= max_levels) {
          printf("max number of refinement levels\n");
          break;
        }
        if (cdofs > max_dofs)
          {
            cout << "Reached the maximum number of dofs. Stop." << endl;
            break;
          }
        
        refiner.Apply(*mesh);
        mesh->Save("mesh_refine.mesh");
        if (refiner.Stop()) {
          cout << "Stopping criterion satisfied. Stop." << endl;
          break;
        } else {
          printf("Refining mesh, i=%d\n", it_amr);
        }

        fespace.Update();

        x.Update();
        xnew.Update();
        res.Update();
        psi_r.Update();
        psi_z.Update();
        Br_field.Update();
        Bp_field.Update();
        Bz_field.Update();
        pv.Update();
        eq_res.Update();
        opt_res.Update();
        b1.Update();
        b3.Update();
        b1p.Update();
        b3p.Update();

        dx_.Update();
        dp_.Update();


        // break;
      }

      if (error < newton_tol) {
        break;
      }
      // break;

      // x += dx.GetBlock(0);
      // pv += dx.GetBlock(1);
      // invHFT->AddMult(dx.GetBlock(1), *uv);
      // invH->AddMult(b2, *uv);
      // if (add_alpha) {
      //   alpha += dx.GetBlock(2)[0];
      //   lv += dx.GetBlock(3)[0];
      // }

      x += dx_;
      pv += dp_;
      invHFT->AddMult(dp_, *uv);
      invH->AddMult(b2, *uv);
      if (add_alpha) {
        alpha += da_;
        lv += dl_;
      }
      

      printf("currents: [");
      for (int i = 0; i < uv->Size(); ++i) {
        printf("%.3e ", (*uv)[i]);
      }
      printf("]\n");
      
      x.Save("gf/xtmp.gf");
      char name[60];
      sprintf(name, "gf/xtmp%d.gf", i);
      x.Save(name);
      Br_field.Save("gf/Br.gf");
      Bp_field.Save("gf/Bp.gf");
      Bz_field.Save("gf/Bz.gf");

      // compute magnetic field
      x.GetDerivative(1, 0, psi_r);
      x.GetDerivative(1, 1, psi_z);
      Br_field.ProjectCoefficient(BrCoeff);
      Bp_field.ProjectCoefficient(BpCoeff);
      Bz_field.ProjectCoefficient(BzCoeff);

      visit_dc.Save();
      

    }
     
  } else {
    // for given currents, solve the GS equations

    GridFunction dx(&fespace);
    dx = 0.0;

    
    LinearForm coil_term(&fespace);
    int ndof__ = fespace.GetNDofs();
    SparseMatrix * F;
    F = new SparseMatrix(ndof__, num_currents);
    DefineRHS(*model, rho_gamma, *mesh, *exact_coefficient, *exact_forcing_coeff, coil_term, F);
  
    BilinearForm diff_operator(&fespace);
    DefineLHS(*model, rho_gamma, diff_operator);

    SparseMatrix * K_;
    Vector g_;
    vector<Vector> *alpha_coeffs;
    vector<Array<int>> *J_inds;
    alpha_coeffs = new vector<Vector>;
    J_inds = new vector<Array<int>>;
    init_coeff->compute_QP(N_control, mesh, &fespace);
    K_ = init_coeff->compute_K();
    g_ = init_coeff->compute_g();
    J_inds = init_coeff->get_J();
    alpha_coeffs = init_coeff->get_alpha();

    SparseMatrix * H;
    H = new SparseMatrix(num_currents, num_currents);
    for (int i = 0; i < num_currents; ++i) {
      if (i < 5) {
        H->Set(i, i, weight_coils);
      } else {
        H->Set(i, i, weight_solenoids);
      }
    }
    H->Finalize();
    
    SysOperator op(&diff_operator, &coil_term, model, &fespace, mesh, attr_lim, &x, F, uv, H, K_, &g_, alpha_coeffs, J_inds, &alpha, include_plasma);
    op.set_i_option(obj_option);
    op.set_obj_weight(obj_weight);
    LinearForm out_vec(&fespace);
    double error_old;
    double error;
    for (int i = 0; i < max_newton_iter; ++i) {

      op.Mult(x, out_vec);
      error = GetMaxError(out_vec);
      // cout << "eq_res" << "i" << i << endl;
      // out_vec.Print();

      printf("\n");
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

      // add(dx, ur_coeff - 1.0, dx, dx);
      x -= dx;

      x.Save("xtmp.gf");
      GridFunction err(&fespace);
      err = out_vec;
      err.Save("res.gf");

      visit_dc.Save();


    }
    op.Mult(x, out_vec);
    error = GetMaxError(out_vec);
    printf("\n\n********************************\n");
    printf("final max residual: %.3e, ratio %.3e\n", error, error_old / error);
    printf("********************************\n\n");
  }
  
}


double gs(const char * mesh_file, const char * data_file, int order, int d_refine,
          int model_choice,
          double & alpha, double & beta, double & gamma, double & mu, double & Ip,
          double & r0, double & rho_gamma, int max_krylov_iter, int max_newton_iter,
          double & krylov_tol, double & newton_tol,
          double & c1, double & c2, double & c3, double & c4, double & c5, double & c6, double & c7,
          double & c8, double & c9, double & c10, double & c11,
          double & ur_coeff,
          int do_control, int N_control, double & weight_solenoids, double & weight_coils,
          double & weight_obj, int obj_option, bool optimize_alpha,
          bool do_manufactured_solution, bool do_initial) {

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
   if (do_initial) {
     mesh.Save("initial.mesh");
   }
   
   // save options in model
   // alpha: multiplier in \bar{S}_{ff'} term
   // beta: multiplier for S_{p'} term
   // gamma: multiplier for S_{ff'} term
   const char *data_file_ = "fpol_pres_ffprim_pprime.data";
   PlasmaModelFile model(mu, data_file_, alpha, beta, gamma, model_choice);

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
      Solve
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------------
    */

   if (do_initial) {
     do_control = false;
   }
   
   // Define the solution x as a finite element grid function in fespace. Set
   // the initial guess to zero, which also sets the boundary conditions.
   GridFunction u(&fespace);
   
   InitialCoefficient init_coeff = read_data_file(data_file);
   if (do_manufactured_solution) {
     u.ProjectCoefficient(exact_coefficient);
     u.Save("exact.gf");
   } else {
     if (!do_initial) {
       ifstream ifs("interpolated.gf");
       GridFunction lgf(&mesh, ifs);
       u = lgf;
     }

     u.Save("initial.gf");

   }

   GridFunction x(&fespace);
   x = u;

   bool include_plasma = true;
   if (do_initial) {
     include_plasma = false;
   }
   
   Solve(fespace, &model, x, kdim, max_newton_iter, max_krylov_iter, newton_tol, krylov_tol, 
         Ip, N_control, do_control,
         optimize_alpha, obj_option, weight_obj,
         rho_gamma,
         &mesh,
         &exact_forcing_coeff,
         &exact_coefficient,
         &init_coeff,
         include_plasma,
         weight_coils,
         weight_solenoids,
         &uv_currents,
         alpha);
   
   if (do_initial) {
     char name_gf_out[60];
     char name_mesh_out[60];
     sprintf(name_gf_out, "initial_guess_g%d.gf", d_refine);
     sprintf(name_mesh_out, "initial_mesh_g%d.mesh", d_refine);

     x.Save(name_gf_out);
     mesh.Save(name_mesh_out);
     printf("Saved solution to %s\n", name_gf_out);
     printf("Saved mesh to %s\n", name_mesh_out);
     printf("glvis -m %s -g %s\n", name_mesh_out, name_gf_out);
   } else {
     x.Save("final.gf");
     printf("Saved solution to final.gf\n");
     printf("Saved mesh to mesh.mesh\n");
     printf("glvis -m mesh.mesh -g final.gf\n");
   }
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
