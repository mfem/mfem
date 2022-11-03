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


void SysOperator::Mult(const Vector &psi, Vector &y) const {
  // diff_operator * psi - plasma_term(psi) * psi - coil_term

  GridFunction x(fespace);
  x = psi;
  int ind_ma, ind_x;
  double val_ma, val_x;
  int iprint = 0;
  set<int> plasma_inds;
  compute_plasma_points(x, *mesh, vertex_map, plasma_inds, ind_ma, ind_x, val_ma, val_x, iprint);
  // val_x = 1.0;
  // val_ma = 0.0;
  NonlinearGridCoefficient nlgcoeff1(model, 1, &x, val_ma, val_x, plasma_inds, attr_lim);
  if ((iprint) || (false)) {
    printf(" val_ma: %f, val_x: %f \n", val_ma, val_x);
    printf(" ind_ma: %d, ind_x: %d \n", ind_ma, ind_x);
  }
  LinearForm plasma_term(fespace);
  plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
  plasma_term.Assemble();

  diff_operator->Mult(psi, y);
  add(y, -1.0, *coil_term, y);
  add(y, -1.0, plasma_term, y);

  // deal with boundary conditions
  Vector u_b_exact, u_tmp, u_b;
  psi.GetSubVector(boundary_dofs, u_b);
  u_tmp = u_b;
  u_boundary->GetSubVector(boundary_dofs, u_b_exact);
  u_tmp -= u_b_exact;
  y.SetSubVector(boundary_dofs, u_tmp);

}




Operator &SysOperator::GetGradient(const Vector &psi) const {
  // diff_operator - sum_{i=1}^3 diff_plasma_term_i(psi)

  delete Mat;
  GridFunction x(fespace);
  x = psi;

  int ind_ma, ind_x;
  double val_ma, val_x;
  int iprint = 0;
  set<int> plasma_inds;
  compute_plasma_points(x, *mesh, vertex_map, plasma_inds, ind_ma, ind_x, val_ma, val_x, iprint);
  // val_x = 1.0;
  // val_ma = 0.0;
  if ((iprint) || (false)) {
    // printf(" val_ma: %f, val_x: %f \n", val_ma, val_x);
    printf(" ind_ma: %d, ind_x: %d \n", ind_ma, ind_x);
  }
  // first nonlinear contribution: bilinear operator
  NonlinearGridCoefficient nlgcoeff_2(model, 2, &x, val_ma, val_x, plasma_inds, attr_lim);
  BilinearForm diff_plasma_term_2(fespace);
  diff_plasma_term_2.AddDomainIntegrator(new MassIntegrator(nlgcoeff_2));
  // diff_plasma_term_2.EliminateEssentialBC(boundary_dofs, DIAG_ZERO);
  diff_plasma_term_2.Assemble();

  // second nonlinear contribution: corresponds to the magnetic axis point column in jacobian
  NonlinearGridCoefficient nlgcoeff_3(model, 3, &x, val_ma, val_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_3(fespace);
  diff_plasma_term_3.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_3));
  diff_plasma_term_3.Assemble();

  // third nonlinear contribution: corresponds to the x-point column in jacobian
  NonlinearGridCoefficient nlgcoeff_4(model, 4, &x, val_ma, val_x, plasma_inds, attr_lim);
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
    // if (diff_plasma_term_3[k] != 0.0) {
    //   printf("%d, %.3e, %.3e\n", k, diff_plasma_term_3[k], diff_plasma_term_4[k]);
    // }
    Mat->Add(k, ind_ma, -diff_plasma_term_3[k]);
    Mat->Add(k, ind_x, -diff_plasma_term_4[k]);
    // note, when no saddles are found, derivative is different?
  }
  Mat->Finalize();

  SparseMatrix *Mat_;
  Mat_ = Add(1.0, M1, -1.0, M2);
  for (int k = 0; k < boundary_dofs.Size(); ++k) {
    Mat_->EliminateRow((boundary_dofs)[k], DIAG_ONE);
  }

  SparseMatrix *Final;
  Final = Add(*Mat_, *Mat);
  
  return *Final;
    
}
