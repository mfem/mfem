#ifndef SYS_OPERATOR
#define SYS_OPERATOR

#include "mfem.hpp"
#include "gs.hpp"
#include "plasma_model.hpp"
using namespace mfem;
using namespace std;

double GetMaxError(LinearForm &res);
double GetMaxError(GridFunction &res);
double GetMaxError(Vector &res);
/*
  Mult:
  diff_operator * psi - plasma_term(psi) * psi - coil_term

  GetGradient:
  diff_operator - sum_{i=1}^3 diff_plasma_term_i(psi)
 */
class SysOperator : public Operator
{
private:
  mutable BilinearForm *diff_operator;
  LinearForm *coil_term;
  mutable PlasmaModelBase *model;
  FiniteElementSpace *fespace;
  Mesh *mesh;
  map<int, vector<int>> vertex_map;
  mutable SparseMatrix *Mat;
  int attr_lim;
  Array<int> boundary_dofs;
  GridFunction *u_boundary;
  bool include_plasma = true;

  GridFunction hat;
  GridFunction ones;

  Vector *uv_currents;
  mutable Vector Plasma_Vec;
  mutable double Alpha_Term;
  mutable double Plasma_Current;
  mutable Vector B_alpha;

  double *alpha_bar;
  int N_control;

  // related to x-point and magnetic axis
  mutable double psi_x;
  mutable double psi_ma;
  mutable double * x_ma;
  mutable double * x_x;
  mutable int ind_ma;
  mutable int ind_x;
  mutable set<int> plasma_inds;

  // related to optimization
  SparseMatrix *F;
  SparseMatrix *H;
  SparseMatrix *K;
  Vector *g;
  vector<Vector> *alpha_coeffs;
  vector<Array<int>> *J_inds;
  int i_option = 0;

  //
  GridFunction res;
  GridFunction Ba;
  GridFunction Cy;
  // Vector res;
  // Vector Ba;
  // Vector Cy;
  SparseMatrix *By;
  double plasma_current;
  double Ca;
  double obj_weight = 1.0;
  
  
public:
  SysOperator(BilinearForm *diff_operator_, LinearForm *coil_term_,
              PlasmaModelBase *model_, FiniteElementSpace *fespace_,
              Mesh *mesh_, int attr_lim_, 
              GridFunction *u_boundary_, SparseMatrix *F_, Vector *uv_currents_,
              SparseMatrix *H_, SparseMatrix *K_, Vector * g_,
              vector<Vector> *alpha_coeffs_,
              vector<Array<int>> *J_inds_,
              double *alpha_bar_,
              bool include_plasma_=true) :
    Operator(fespace_->GetTrueVSize()), diff_operator(diff_operator_),
    coil_term(coil_term_), model(model_), fespace(fespace_),
    mesh(mesh_), Mat(NULL), attr_lim(attr_lim_),
    u_boundary(u_boundary_),
    F(F_), uv_currents(uv_currents_), H(H_), K(K_), g(g_),
    alpha_coeffs(alpha_coeffs_), J_inds(J_inds_),
    alpha_bar(alpha_bar_),
    include_plasma(include_plasma_)
  {
    if (alpha_coeffs != NULL) {
      N_control = alpha_coeffs->size();
    }
    vertex_map = compute_vertex_map(*mesh, attr_lim);
    // vertex_map = compute_vertex_map(*mesh, -1);

    // Extract the list of all the boundary DOFs.
    // The r=0 boundary will be marked as dirichlet (psi=0)
    // and the far-field will not be marked as dirichlet
    Array<int> bdr_attribs(mesh->bdr_attributes);
    Array<int> ess_bdr(bdr_attribs.Max());
    ess_bdr = 0;
    // ess_bdr[attr_ff_bdr-1] = 0;
    int attr_axis = 900;
    ess_bdr[attr_axis-1] = 1;
    int box_axis = 832;
    ess_bdr[box_axis-1] = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, boundary_dofs, 1);

    Vector pw_vector(3000);
    pw_vector = 1.0;
    pw_vector(1100-1) = 0.0;
    PWConstCoefficient pw_coeff(pw_vector);
    GridFunction hat_(fespace);
    hat_.ProjectCoefficient(pw_coeff);
    hat = hat_;
    // hat.Save("hat.gf");

    GridFunction ones_(fespace);
    ones_ = 1.0;
    ones = ones_;

    int vsize = fespace->GetTrueVSize();

    // Vector res_(vsize);
    GridFunction res_(fespace);
    res_ = 0.0;
    res = res_;

    // Vector Ba_(vsize);
    GridFunction Ba_(fespace);
    Ba_ = 0.0;
    Ba = Ba_;

    // Vector Cy_(vsize);
    GridFunction Cy_(fespace);
    Cy_ = 0.0;
    Cy = Cy_;
    // GridFunction Cy_(fespace);
    // Cy_ = 0.0;
    // Cy = Cy_;

    // int m = fespace->GetTrueVSize();
    // SparseMatrix By_(m, m);
    // By = By_;
    
    
  }
  virtual void Mult(const Vector &psi, Vector &y) const;
  virtual Operator &GetGradient(const Vector &psi) const;
  virtual ~SysOperator() { };

  void compute_psi_N_derivs(const GridFunction &psi, int k, double * psi_N,
                            GridFunction * grad_psi_N, SparseMatrix * hess_psi_N);

  double compute_obj(const GridFunction &psi);
  GridFunction compute_grad_obj(const GridFunction &psi);
  SparseMatrix* compute_hess_obj(const GridFunction &psi);

  double get_mu() {
    return model->get_mu();
  }
  SparseMatrix * GetF() {
    return F;
  }
  SparseMatrix * GetK() {
    return K;
  }
  SparseMatrix * GetH() {
    return H;
  }
  Vector * get_g() {
    return g;
  }
  Vector * get_uv() {
    return uv_currents;
  }
  int get_N_control() {
    return N_control;
  }
  double *get_alpha_bar() {
    return alpha_bar;
  }
  double get_Alpha_Term() {
    return Alpha_Term;
  }
  Vector get_B_alpha() {
    return B_alpha;
  }
  double get_Plasma_Current() {
    return Plasma_Current;
  }
  Vector get_Plasma_Vec() {
    return Plasma_Vec;
  }
  double get_psi_x() {
    return psi_x;
  }
  double get_psi_ma() {
    return psi_ma;
  }
  int get_ind_x() {
    return ind_x;
  }
  int get_ind_ma() {
    return ind_ma;
  }
  double* get_x_x() {
    return x_x;
  }
  double* get_x_ma() {
    return x_ma;
  }

  Vector get_res() {return res;}
  // GridFunction* get_res() {return res;}
  SparseMatrix get_By() {return *By;}
  Vector get_Ba() {return Ba;}
  double get_plasma_current() {return plasma_current;}
  Vector get_Cy() {return Cy;}
  double get_Ca() {return Ca;}

  void set_i_option(int option) {i_option = option;}
  void set_obj_weight(double obj_weight_) {obj_weight = obj_weight_;}

  double get_plasma_current(GridFunction &x, double &alpha);

  void NonlinearEquationRes(GridFunction &x, Vector *currents, double &alpha);
    
};


/*
  By - 1/Ca Ba Cy^T
 */
class ByMinusRankOnePerturbation : public Operator {
private:
  const SparseMatrix *By;
  const Vector *Ba, *Cy;
  const double ca;

public:
  //set parameters
  ByMinusRankOnePerturbation(SparseMatrix *By_, const Vector  *Ba_, const Vector  *Cy_, double ca_) :
    By(By_), Ba(Ba_), Cy(Cy_), ca(ca_)
  {
    width = Ba_->Size();
    height = Ba_->Size();
  }

  virtual void Mult(const Vector &k, Vector &y) const;
  virtual void MultTranspose(const Vector &k, Vector &y) const;

  virtual ~ByMinusRankOnePerturbation() {}
};


// D - C A^{-1} B
class SchurComplement : public Operator {
private:
  const Operator *D;
  const Operator *C;
  const Operator *B;
  const Operator *A;
  Solver *A_prec;
  GMRESSolver solver;
  mutable Vector a, b;
  
public:
  SchurComplement(Operator *A_, Operator *B_, Operator *C_, Operator *D_, Solver *A_prec_) :
    A(A_), B(B_), C(C_), D(D_), A_prec(A_prec_)
  {

    width = A->Height();
    height = A->Height();

    double krylov_tol = 1e-12;
    double krylov_tol_light = 1e-12;
    int max_krylov_iter = 1000;
    int kdim = 1000;
    
    solver.SetAbsTol(0.0);
    solver.SetRelTol(krylov_tol);
    solver.SetMaxIter(max_krylov_iter);
    solver.SetOperator(*A);
    solver.SetPreconditioner(*A_prec);
    solver.SetKDim(kdim);
    solver.SetPrintLevel(0);
    
    Vector a_(A->Height());
    Vector b_(A->Height());
    a_ = 0.0;
    b_ = 0.0;
    b = b_;
    a = a_;

  }

  virtual void Mult(const Vector &k, Vector &y) const;

  virtual ~SchurComplement() {}
};



// (D - C A^{-1} B)^{-1}
class SchurComplementInverse : public Operator {
private:
  const Operator *SC;
  Solver *SC_prec;
  GMRESSolver solver;
  mutable Vector a, b;
  
public:
  SchurComplementInverse(SchurComplement *SC_, Solver *SC_prec_) :
    SC(SC_), SC_prec(SC_prec_)
  {

    width = SC->Height();
    height = SC->Height();

    double krylov_tol = 1e-12;
    double krylov_tol_light = 1e-12;
    int max_krylov_iter = 1000;
    int kdim = 1000;
    
    solver.SetAbsTol(0.0);
    solver.SetRelTol(krylov_tol);
    solver.SetMaxIter(max_krylov_iter);
    solver.SetOperator(*SC_);
    solver.SetPreconditioner(*SC_prec_);
    solver.SetKDim(kdim);
    solver.SetPrintLevel(0);
    
    Vector a_(SC->Height());
    Vector b_(SC->Height());
    a_ = 0.0;
    b_ = 0.0;
    b = b_;
    a = a_;

  }

  virtual void Mult(const Vector &k, Vector &y) const;

  virtual ~SchurComplementInverse() {}
};




// /*
//   -F H^{-1} F^T
//  */
// class mMuFinvHFT : public Operator{
// private:
//   const SparseMatrix *F;
//   const SparseMatrix *invH;

// public:
//   //set parameters
//   mFinvHFT(SparseMatrix *F_, SparseMatrix *invH_) :
//     F(F_), invH(invH_)
//   {
//     width = F_->Height();
//     height = F_->Height();
//   }

//   // set action
//   virtual void Mult(const Vector &k, Vector &y) const;

//   virtual ~mFinvHFT() {}
// };


class WoodburyInverse : public Operator {
private:
  SparseMatrix *A;
  Solver *A_prec;
  const Vector *U, *V;
  const double scale;
  GMRESSolver solver;
  mutable Vector a, b;
  double dot;
public:
  // set parameters
  WoodburyInverse(SparseMatrix *A_, Solver *A_prec_, Vector *U_, Vector *V_, double scale_) :
    A(A_), A_prec(A_prec_), U(U_), V(V_), scale(scale_)
  {
    width = U_->Size();
    height = U_->Size();

    double krylov_tol = 1e-12;
    double krylov_tol_light = 1e-12;
    int max_krylov_iter = 1000;
    int kdim = 1000;

    Vector a_(U_->Size());
    Vector b_(U_->Size());
    a_ = 0.0;
    b_ = 0.0;

    solver.SetAbsTol(0.0);
    solver.SetRelTol(krylov_tol);
    solver.SetMaxIter(max_krylov_iter);
    solver.SetOperator(*A);
    solver.SetPreconditioner(*A_prec);
    solver.SetKDim(kdim);
    solver.SetPrintLevel(0);

    solver.Mult(*U, b_);

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

    b = b_;
    a = a_;

    solver.SetRelTol(krylov_tol_light);

    dot = (*V) * b;
  }

  // set action
  virtual void Mult(const Vector &k, Vector &y) const;

  virtual ~WoodburyInverse() {}
};



#endif
