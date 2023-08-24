#include "mfem.hpp"
#include "sys_operator.hpp"
using namespace mfem;
using namespace std;


double GetMaxError(LinearForm &res) {
  double *resv = res.GetData();
  int size = res.FESpace()->GetTrueVSize();

  double max_val = - numeric_limits<double>::infinity();
  for (int i = 0; i < size; ++i) {
    if (abs(resv[i]) > max_val) {
      max_val = abs(resv[i]);
    }
  }
  return max_val;
}

double GetMaxError(GridFunction &res) {
  double *resv = res.GetData();
  int size = res.FESpace()->GetTrueVSize();

  double max_val = - numeric_limits<double>::infinity();
  for (int i = 0; i < size; ++i) {
    if (abs(resv[i]) > max_val) {
      max_val = abs(resv[i]);
    }
  }
  return max_val;
}

double GetMaxError(Vector &res) {
  int size = res.Size();

  double max_val = - numeric_limits<double>::infinity();
  for (int i = 0; i < size; ++i) {
    if (abs(res[i]) > max_val) {
      max_val = abs(res[i]);
    }
  }
  return max_val;
}

void Print(const Vector &y) {
  for (int i = 0; i < y.Size(); ++i) {
    printf("%d %.14e\n", i+1, y[i]);
  }
}

void Print(const LinearForm &y) {
  double *yv = y.GetData();
  int size = y.FESpace()->GetTrueVSize();
  for (int i = 0; i < size; ++i) {
    printf("%d %.14e\n", i+1, yv[i]);
  }
}



void SysOperator::compute_psi_N_derivs(const GridFunction &psi, int k, double * psi_N,
                                       GridFunction * grad_psi_N,
                                       SparseMatrix * hess_psi_N) {

  int dof = (*alpha_coeffs)[0].Size();
  double psi_interp;
  double psi_x = psi[ind_x];
  double psi_ma = psi[ind_ma];
  
  psi_interp = 0;
  for (int m = 0; m < dof; ++m) {
    int i = (*J_inds)[k][m];
    psi_interp += (*alpha_coeffs)[k][m] * psi[i];
  }

  (*psi_N) = (psi_interp - psi_ma) / (psi_x - psi_ma);

  for (int m = 0; m < dof; ++m) {
    int i = (*J_inds)[k][m];
    (*grad_psi_N)[i] += (*alpha_coeffs)[k][m] / (psi_x - psi_ma);
  }
  (*grad_psi_N)[ind_ma] += ((*psi_N) - 1.0) / (psi_x - psi_ma);
  (*grad_psi_N)[ind_x] += - (*psi_N) / (psi_x - psi_ma);

  int i, j;
  for (int m = 0; m < dof+2; ++m) {
    if (m < dof) {
      i = (*J_inds)[k][m];
    } else if (m == dof) {
      i = ind_ma;
    } else {
      i = ind_x;
    }

    for (int n = 0; n < dof+2; ++n) {
      if (n < dof) {
        j = (*J_inds)[k][n];
      } else if (n == dof) {
        j = ind_ma;
      } else {
        j = ind_x;
      }

      hess_psi_N->Add(i, j, 0.0);
      if (j == ind_x) {
        hess_psi_N->Add(i, j, - (*grad_psi_N)[i] / (psi_x - psi_ma));
      }
      if (j == ind_ma) {
        hess_psi_N->Add(i, j, (*grad_psi_N)[i] / (psi_x - psi_ma));
      }
      if (i == ind_x) {
        hess_psi_N->Add(i, j, - (*grad_psi_N)[j] / (psi_x - psi_ma));
      }
      if (i == ind_ma) {
        hess_psi_N->Add(i, j, (*grad_psi_N)[j] / (psi_x - psi_ma));
      }
    }
  }
  
  hess_psi_N->Finalize(0);
}


double SysOperator::compute_obj(const GridFunction &psi) {

  int dof = (*alpha_coeffs)[0].Size();
  double psi_interp, psi_interp_0, psi_N;
  double obj = 0;
  double psi_x = psi[ind_x];
  double psi_ma = psi[ind_ma];
  double weight = obj_weight;

  if (i_option == 0) {
    // obj = sum_{n=1}^{N} (psi - psi_0) ^ 2

    int k = 0;
    psi_interp_0 = 0;
    for (int m = 0; m < dof; ++m) {
      int i = (*J_inds)[k][m];
      psi_interp_0 += (*alpha_coeffs)[k][m] * psi[i];
    }
      
    for (int k = 1; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      obj += 0.5 * (psi_interp - psi_interp_0) * (psi_interp - psi_interp_0);

    }

  } else if (i_option == 1) {
    // obj = sum_{n=1}^{N} (psi_N - 1) ^ 2

    for (int k = 0; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      psi_N = (psi_interp - psi_ma) / (psi_x - psi_ma);
      obj += 0.5 * (psi_N - 1.0) * (psi_N - 1.0);
    }
  } else {
    // obj = sum_{n=1}^{N} (psi - psi_x) ^ 2

    for (int k = 0; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      obj += 0.5 * (psi_interp - psi_x) * (psi_interp - psi_x);
    }
  }

  obj *= weight;
  return obj;
}


GridFunction SysOperator::compute_grad_obj(const GridFunction &psi) {
  
  double psi_x = psi[ind_x];
  double psi_ma = psi[ind_ma];
  FiniteElementSpace fespace = *(psi.FESpace());
  // int ndof = psi.Size();
  int ndof = fespace.GetNDofs();
  double weight = obj_weight;
  
  GridFunction g_(&fespace);
  g_ = 0.0;
  if (i_option == 0) {
    K->Mult(psi, g_);

  } else if (i_option == 1) {
    int dof = (*alpha_coeffs)[0].Size();
    double psi_N;

    GridFunction * grad_psi_N;
    grad_psi_N = new GridFunction(&fespace);
    SparseMatrix * hess_psi_N;
    for (int k = 0; k < N_control; ++k) {

      hess_psi_N = new SparseMatrix(ndof, ndof);

      *grad_psi_N = 0.0;
      compute_psi_N_derivs(psi, k, &psi_N, grad_psi_N, hess_psi_N);

      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        if ((i == ind_ma) || (i == ind_x)) {
          continue;
        }
        g_[i] += (psi_N - 1.0) * (*grad_psi_N)[i];
      }

      g_[ind_ma] += (psi_N - 1.0) * (*grad_psi_N)[ind_ma];
      g_[ind_x] += (psi_N - 1.0) * (*grad_psi_N)[ind_x];
    }
    delete hess_psi_N;


    return g_;
  } else {
    int dof = (*alpha_coeffs)[0].Size();
    double psi_interp;
    for (int k = 0; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        g_[i] += (psi_interp - psi_x) * (*alpha_coeffs)[k][m];
      }
      g_[ind_x] += - (psi_interp - psi_x);
    }

  }

  g_ *= weight;
  return g_;
  
}



SparseMatrix* SysOperator::compute_hess_obj(const GridFunction &psi) {

  double psi_x = psi[ind_x];
  double psi_ma = psi[ind_ma];
  // int ndof = psi.Size();
  FiniteElementSpace fespace = *(psi.FESpace());
  int ndof = fespace.GetNDofs();
  double weight = obj_weight;

  if (i_option == 0) {
    *K *= weight;
    return K;
  } else if (i_option == 1) {
  
    SparseMatrix * K_;
    K_ = new SparseMatrix(ndof, ndof);

    int dof = (*alpha_coeffs)[0].Size();
    double psi_N;

    GridFunction * grad_psi_N;
    grad_psi_N = new GridFunction(&fespace);
      
    SparseMatrix * hess_psi_N;

    int i, j;
    for (int k = 0; k < N_control; ++k) {

      hess_psi_N = new SparseMatrix(ndof, ndof);
      
      *grad_psi_N = 0.0;
      compute_psi_N_derivs(psi, k, &psi_N, grad_psi_N, hess_psi_N);

      for (int m = 0; m < dof+2; ++m) {
        if (m < dof) {
          i = (*J_inds)[k][m];
          if ((i == ind_ma) || (i == ind_x)) {
            continue;
          }
        } else if (m == dof) {
          i = ind_ma;
        } else {
          i = ind_x;
        }

        for (int n = 0; n < dof+2; ++n) {
          if (n < dof) {
            j = (*J_inds)[k][n];
            if ((j == ind_ma) || (j == ind_x)) {
              continue;
            }
          } else if (n == dof) {
            j = ind_ma;
          } else {
            j = ind_x;
          }
          K_->Add(i, j, (*grad_psi_N)[i] * (*grad_psi_N)[j] + (*hess_psi_N)(i, j) * (psi_N - 1));
        }
      }
      delete hess_psi_N;

    }
    K_->Finalize();
    *K_ *= weight;
    return K_;
  } else {
    SparseMatrix * K_;
    K_ = new SparseMatrix(ndof, ndof);
    int dof = (*alpha_coeffs)[0].Size();
    int i, j;
    double a, b;
    for (int k = 0; k < N_control; ++k) {

      for (int m = 0; m < dof+1; ++m) {
        if (m < dof) {
          i = (*J_inds)[k][m];
          a = (*alpha_coeffs)[k][m];
        } else {
          i = ind_x;
          a = -1.0;
        }

        for (int n = 0; n < dof+1; ++n) {
          if (n < dof) {
            j = (*J_inds)[k][n];
            b = (*alpha_coeffs)[k][n];
          } else {
            j = ind_x;
            b = -1.0;
          }
          
          K_->Add(i, j, a * b);
          // K_->Add(i, j, (*alpha_coeffs)[k][m] * (*alpha_coeffs)[k][n]);
          
        }
      }

    }
    
    K_->Finalize();
    *K_ *= weight;
    return K_;
  }
  
}


double SysOperator::get_plasma_current(GridFunction &x, double &alpha) {

  model->set_alpha_bar(alpha);

  double val_ma, val_x;
  int iprint = 0;
  set<int> plasma_inds_;
  compute_plasma_points(&x, *mesh, vertex_map, plasma_inds_, ind_ma, ind_x, val_ma, val_x, iprint);
  psi_x = val_x;
  psi_ma = val_ma;
  plasma_inds = plasma_inds_;

  double* x_ma_ = mesh->GetVertex(ind_ma);
  double* x_x_ = mesh->GetVertex(ind_x);
  x_ma = x_ma_;
  x_x = x_x_;

  NonlinearGridCoefficient nlgcoeff1(model, 1, &x, val_ma, val_x, plasma_inds, attr_lim);

  LinearForm plasma_term(fespace);
  plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
  plasma_term.Assemble();

  Plasma_Current = plasma_term(ones);
  Plasma_Current *= -1.0;

  return Plasma_Current;
}










void SysOperator::NonlinearEquationRes(GridFunction &psi, Vector *currents, double &alpha) {

  GridFunction x(fespace);
  x = psi;

  // set alpha in the plasma model
  model->set_alpha_bar(alpha);

  // compute the x point and magnetic axis
  double val_ma, val_x;
  int iprint = 0;
  set<int> plasma_inds_;
  compute_plasma_points(&x, *mesh, vertex_map, plasma_inds_, ind_ma, ind_x, val_ma, val_x, iprint);
  psi_x = val_x;
  psi_ma = val_ma;
  plasma_inds = plasma_inds_;
  double* x_ma_ = mesh->GetVertex(ind_ma);
  double* x_x_ = mesh->GetVertex(ind_x);
  x_ma = x_ma_;
  x_x = x_x_;

  // ------------------------------------------------
  // *** compute res ***
  // contribution from plasma terms
  NonlinearGridCoefficient nlgcoeff1(model, 1, &x, val_ma, val_x, plasma_inds, attr_lim);
  LinearForm plasma_term(fespace);
  plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
  plasma_term.Assemble();

  // contribution from LHS
  diff_operator->Mult(psi, res);
  add(res, -1.0, *coil_term, res);
  if (include_plasma) {
    add(res, -1.0, plasma_term, res);
  }

  // contribution from currents
  F->AddMult(*currents, res, -model->get_mu());

  // boundary conditions
  Vector u_b_exact, u_tmp, u_b;
  psi.GetSubVector(boundary_dofs, u_b);
  u_tmp = u_b;
  u_boundary->GetSubVector(boundary_dofs, u_b_exact);
  u_tmp -= u_b_exact;
  res.SetSubVector(boundary_dofs, u_tmp);

  // zero-out residual where we are interpolating from a guess
  res *= hat;

  // ------------------------------------------------
  // *** compute jacobian ***

  // first nonlinear contribution: bilinear operator
  NonlinearGridCoefficient nlgcoeff_2(model, 2, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  BilinearForm diff_plasma_term_2(fespace);
  diff_plasma_term_2.AddDomainIntegrator(new MassIntegrator(nlgcoeff_2));
  diff_plasma_term_2.Assemble();

  // second nonlinear contribution: corresponds to the magnetic axis point column in jacobian
  NonlinearGridCoefficient nlgcoeff_3(model, 3, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_3(fespace);
  diff_plasma_term_3.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_3));
  diff_plasma_term_3.Assemble();

  // third nonlinear contribution: corresponds to the x-point column in jacobian
  NonlinearGridCoefficient nlgcoeff_4(model, 4, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_4(fespace);
  diff_plasma_term_4.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_4));
  diff_plasma_term_4.Assemble();

  // turn diff_operator and diff_plasma_term_2 into sparse matrices
  SparseMatrix diff_operator_sp_mat = diff_operator->SpMat();
  SparseMatrix psi_coeff_sp_mat = diff_plasma_term_2.SpMat();

  diff_operator_sp_mat.Finalize();
  psi_coeff_sp_mat.Finalize();

  // create a new sparse matrix, Mat, that will combine all terms
  int m = fespace->GetTrueVSize();
  SparseMatrix *psi_x_psi_ma_coeff_sp_mat;
  psi_x_psi_ma_coeff_sp_mat = new SparseMatrix(m, m);
  for (int k = 0; k < m; ++k) {
    psi_x_psi_ma_coeff_sp_mat->Add(k, ind_ma, -diff_plasma_term_3[k]);
    psi_x_psi_ma_coeff_sp_mat->Add(k, ind_x, -diff_plasma_term_4[k]);
  }
  psi_x_psi_ma_coeff_sp_mat->Finalize();

  // psi_x_psi_ma_coeff_sp_mat->PrintMatlab();

  SparseMatrix *Mat_Prelim;
  if (!include_plasma) {
    Mat_Prelim = Add(1.0, diff_operator_sp_mat, 0.0, diff_operator_sp_mat);
  } else {
    Mat_Prelim = Add(1.0, diff_operator_sp_mat, -1.0, psi_coeff_sp_mat);
  }
  for (int k = 0; k < boundary_dofs.Size(); ++k) {
    Mat_Prelim->EliminateRow((boundary_dofs)[k], DIAG_ONE);
  }

  if (!include_plasma) {
    By = Add(1.0, *Mat_Prelim, 0.0, *Mat_Prelim);
  } else {
    By = Add(*Mat_Prelim, *psi_x_psi_ma_coeff_sp_mat);
  }
  // By->Finalize();

  // printf("----------*****----\n");
  // By->PrintMatlab();
  // ------------------------------------------------
  // *** compute B_alpha ***
  
  // derivative with respect to alpha
  NonlinearGridCoefficient nlgcoeff_5(model, 5, &x, psi_ma, psi_x, plasma_inds, attr_lim);

  // int_{Omega} 1 / (mu r) \frac{d \bar{S}_{ff'}}{da} v dr dz
  LinearForm diff_plasma_term_5(fespace);
  diff_plasma_term_5.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_5));
  diff_plasma_term_5.Assemble();

  Ba = diff_plasma_term_5;
  Ba *= -1.0;

  // ------------------------------------------------
  // *** compute plasma current ***
  // Ip = int ...
  // d_\psi ...
  SparseMatrix *Mat_Plasma;
  Mat_Plasma = Add(-1.0, *psi_x_psi_ma_coeff_sp_mat, 1.0, psi_coeff_sp_mat);

  // GridFunction ones(fespace);
  // ones = 1.0;
  plasma_current = plasma_term(ones);
  plasma_current *= -1.0;
  
  // ------------------------------------------------
  // *** compute Cy ***
  Vector Plasma_Vec_(m);
  Mat_Plasma->MultTranspose(ones, Plasma_Vec_);
  Plasma_Vec_ *= -1.0;
  Cy = Plasma_Vec_;

  // ------------------------------------------------
  // *** compute Ca ***
  // - int_{Omega_p} 1 / (mu r) \frac{d \bar{S}_{ff'}}{da} dr dz
  Ca = diff_plasma_term_5(ones);
  Ca *= -1.0;

  // printf("%e\n", Ca);



}


////////////////////////////////////////////////////////////////
void SysOperator::Mult(const Vector &psi, Vector &y) const {
  // diff_operator * psi - plasma_term(psi) * psi - coil_term
  
  GridFunction x(fespace);
  x = psi;
  // x.Save("x.gf");
  model->set_alpha_bar(*alpha_bar);

  double val_ma, val_x;
  int iprint = 0;
  set<int> plasma_inds_;
  compute_plasma_points(&x, *mesh, vertex_map, plasma_inds_, ind_ma, ind_x, val_ma, val_x, iprint);
  psi_x = val_x;
  psi_ma = val_ma;
  plasma_inds = plasma_inds_;

  double* x_ma_ = mesh->GetVertex(ind_ma);
  double* x_x_ = mesh->GetVertex(ind_x);
  x_ma = x_ma_;
  x_x = x_x_;
  
  NonlinearGridCoefficient nlgcoeff1(model, 1, &x, val_ma, val_x, plasma_inds, attr_lim);
  if ((iprint) || (false)) {
    printf(" val_ma: %f, val_x: %f \n", val_ma, val_x);
    printf(" ind_ma: %d, ind_x: %d \n", ind_ma, ind_x);
  }

  // GridFunction pt(fespace);
  // pt.ProjectCoefficient(nlgcoeff1);
  // pt.Save("pt.gf");
  
  LinearForm plasma_term(fespace);
  plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
  plasma_term.Assemble();

  Plasma_Current = plasma_term(ones);
  Plasma_Current *= -1.0;
  
  // note: coil term no longer includes current contributions
  // that is included in F matrix below...
  diff_operator->Mult(psi, y);
  add(y, -1.0, *coil_term, y);
  if (include_plasma) {
    add(y, -1.0, plasma_term, y);
  }

  double weight = 1.0;
  // F->AddMult(*uv_currents, y, -1.0 / weight);
  F->AddMult(*uv_currents, y, -model->get_mu());

  // deal with boundary conditions
  Vector u_b_exact, u_tmp, u_b;
  psi.GetSubVector(boundary_dofs, u_b);
  u_tmp = u_b;
  u_boundary->GetSubVector(boundary_dofs, u_b_exact);
  u_tmp -= u_b_exact;
  y.SetSubVector(boundary_dofs, u_tmp);

  for (int k = 0; k < psi.Size(); ++k) {
    y[k] *= hat[k];
  }
}




Operator &SysOperator::GetGradient(const Vector &psi) const {
  // diff_operator - sum_{i=1}^3 diff_plasma_term_i(psi)

  delete Mat;
  GridFunction x(fespace);
  x = psi;
  model->set_alpha_bar(*alpha_bar);

  // first nonlinear contribution: bilinear operator
  NonlinearGridCoefficient nlgcoeff_2(model, 2, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  BilinearForm diff_plasma_term_2(fespace);
  diff_plasma_term_2.AddDomainIntegrator(new MassIntegrator(nlgcoeff_2));
  // diff_plasma_term_2.EliminateEssentialBC(boundary_dofs, DIAG_ZERO);
  diff_plasma_term_2.Assemble();

  // second nonlinear contribution: corresponds to the magnetic axis point column in jacobian
  NonlinearGridCoefficient nlgcoeff_3(model, 3, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_3(fespace);
  diff_plasma_term_3.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_3));
  diff_plasma_term_3.Assemble();

  // third nonlinear contribution: corresponds to the x-point column in jacobian
  NonlinearGridCoefficient nlgcoeff_4(model, 4, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_4(fespace);
  diff_plasma_term_4.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_4));
  diff_plasma_term_4.Assemble();

  // turn diff_operator and diff_plasma_term_2 into sparse matrices
  SparseMatrix M1 = diff_operator->SpMat();
  SparseMatrix M2 = diff_plasma_term_2.SpMat();

  M1.Finalize();
  M2.Finalize();

  // create a new sparse matrix, Mat, that will combine all terms
  int m = fespace->GetTrueVSize();

  // 
  Mat = new SparseMatrix(m, m);
  for (int k = 0; k < m; ++k) {
    Mat->Add(k, ind_ma, -diff_plasma_term_3[k]);
    Mat->Add(k, ind_x, -diff_plasma_term_4[k]);
    // note, when no saddles are found, derivative is different?
  }
  Mat->Finalize();

  SparseMatrix *Mat_;
  if (!include_plasma) {
    Mat_ = Add(1.0, M1, 0.0, M1);
  } else {
    Mat_ = Add(1.0, M1, -1.0, M2);
  }
  for (int k = 0; k < boundary_dofs.Size(); ++k) {
    Mat_->EliminateRow((boundary_dofs)[k], DIAG_ONE);
  }

  SparseMatrix *Final;
  if (!include_plasma) {
    Final = Add(1.0, *Mat_, 0.0, *Mat_);
  } else {
    Final = Add(*Mat_, *Mat);
  }

  // what is this?
  // for (int k = 0; k < psi.Size(); ++k) {
  //   if (abs(psi[k]) > 1e-6) {
  //     Final->EliminateRow(k, DIAG_ONE);
  //   }
  // }
  
  // Ip = int ...
  // d_\psi ...
  SparseMatrix *Mat_Plasma;
  Mat_Plasma = Add(-1.0, *Mat, 1.0, M2);
  GridFunction ones(fespace);
  ones = 1.0;
  Vector Plasma_Vec_(m);
  Mat_Plasma->MultTranspose(ones, Plasma_Vec_);
  Plasma_Vec_ *= -1.0;
  Plasma_Vec = Plasma_Vec_;

  // derivative with respect to alpha
  NonlinearGridCoefficient nlgcoeff_5(model, 5, &x, psi_ma, psi_x, plasma_inds, attr_lim);

  // int_{Omega} 1 / (mu r) \frac{d \bar{S}_{ff'}}{da} v dr dz
  LinearForm diff_plasma_term_5(fespace);
  diff_plasma_term_5.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_5));
  diff_plasma_term_5.Assemble();

  B_alpha = diff_plasma_term_5;
  B_alpha *= -1.0;

  // - int_{Omega_p} 1 / (mu r) \frac{d \bar{S}_{ff'}}{da} dr dz
  Alpha_Term = diff_plasma_term_5(ones);
  Alpha_Term *= -1.0;
  
  return *Final;
    
}
