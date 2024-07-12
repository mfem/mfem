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
  GridFunction f;
  SparseMatrix *By;
  SparseMatrix *By_symmetric;
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
    alpha_bar(alpha_bar_), ones(fespace), hat(fespace),
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
  SparseMatrix get_By_symmetric() {return *By_symmetric;}
  Vector get_Ba() {return Ba;}
  double get_plasma_current() {return plasma_current;}
  Vector get_Cy() {return Cy;}
  double get_Ca() {return Ca;}
  GridFunction get_f() {return f;}

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
  // GMRESSolver solver;
  CGSolver solver;
  mutable Vector a, b;
  mutable int total_iterations=0;
  mutable int total_calls=0;
  
public:
  SchurComplement(Operator *A_, Operator *B_, Operator *C_, Operator *D_, Solver *A_prec_, double krylov_tol=1e-5) :
    A(A_), B(B_), C(C_), D(D_), A_prec(A_prec_)
  {

    width = A->Height();
    height = A->Height();

    int max_krylov_iter = 1000;
    int kdim = 1000;

    solver.SetAbsTol(0.0);
    solver.SetRelTol(krylov_tol);
    solver.SetMaxIter(max_krylov_iter);
    solver.SetOperator(*A);
    solver.SetPreconditioner(*A_prec);
    // solver.SetKDim(kdim);
    solver.SetPrintLevel(-1);
    // solver.SetMaxIter(40);

    Vector a_(A->Height());
    Vector b_(A->Height());
    a_ = 0.0;
    b_ = 0.0;
    b = b_;
    a = a_;

  }

  double GetAvgIterations() const {
    return ((double) total_iterations) / ((double) total_calls);
  }

  double GetTotalIterations() const {
    return total_iterations;
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
  mutable int total_iterations=0;
  mutable int total_calls=0;
  
public:
  SchurComplementInverse(SchurComplement *SC_, Solver *SC_prec_, double krylov_tol=1e-5) :
    SC(SC_), SC_prec(SC_prec_)
  {

    width = SC->Height();
    height = SC->Height();

    int max_krylov_iter = 1000;
    int kdim = 1000;
    
    solver.SetAbsTol(0.0);
    solver.SetRelTol(krylov_tol);
    solver.SetMaxIter(max_krylov_iter);
    solver.SetOperator(*SC_);
    solver.SetPreconditioner(*SC_prec_);
    solver.SetKDim(kdim);
    solver.SetPrintLevel(-1);
    // solver.SetMaxIter(40);
    
    Vector a_(SC->Height());
    Vector b_(SC->Height());
    a_ = 0.0;
    b_ = 0.0;
    b = b_;
    a = a_;

  }
  double GetAvgIterations() const {
    return ((double) total_iterations) / ((double) total_calls);
  }

  double GetTotalIterations() const {
    return total_iterations;
  }
  
  virtual void Mult(const Vector &k, Vector &y) const;

  virtual ~SchurComplementInverse() {}
};


class WoodburyInverse : public Operator {
private:

  SparseMatrix *A;
  Solver *A_prec;
  const Vector *U, *V;
  const double scale;
  GMRESSolver solver;
  mutable Vector a, b;
  double dot;

  mutable int total_iterations=0;
  mutable int total_calls=0;
  
public:
  // set parameters
  WoodburyInverse(SparseMatrix *A_, Solver *A_prec_, Vector *U_, Vector *V_, double scale_, double krylov_tol=1e-5) :
    A(A_), A_prec(A_prec_), U(U_), V(V_), scale(scale_)
  {
    width = U_->Size();
    height = U_->Size();

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
    solver.SetPrintLevel(-1);

    solver.Mult(*U, b_);

    total_calls += 1;
    total_iterations += solver.GetNumIterations();

    b = b_;
    a = a_;

    solver.SetRelTol(krylov_tol);

    dot = (*V) * b;
  }

  double GetAvgIterations() const {
    return ((double) total_iterations) / ((double) total_calls);
  }

  // set action
  virtual void Mult(const Vector &k, Vector &y) const;

  virtual ~WoodburyInverse() {}
};


class WoodburyInversePC : public Solver {
private:
  Solver *A_prec;
  const Vector *U, *V;
  const double scale;
  mutable Vector a, b;
  double dot;

public:
  // set parameters
  WoodburyInversePC(Solver *A_prec_, Vector *U_, Vector *V_, double scale_) :
    A_prec(A_prec_), U(U_), V(V_), scale(scale_)
  {
    width = U_->Size();
    height = U_->Size();

    Vector a_(U_->Size());
    Vector b_(U_->Size());
    a_ = 0.0;
    b_ = 0.0;

    A_prec->Mult(*U, b_);

    b = b_;
    a = a_;

    dot = (*V) * b;
  }

  // set action
  virtual void Mult(const Vector &k, Vector &y) const;
  virtual void SetOperator(const Operator &op) {};
  virtual ~WoodburyInversePC() {}
};


class SchurPC : public Solver {
private:
  Solver *B_prec;
  Solver *BT_prec;
  Operator *A;
  Operator *C;
  Array<int> *dofs1, *dofs2;
  Vector *vec1, *vec2;
  Array<int> *offsets;
  int N;
  int option = 1;
  Vector *Ba;
  Vector *Cy;
  double Ca;

public:
  // set parameters
  SchurPC(Operator *A_, Operator *C_, Solver *B_prec_, Solver *BT_prec_,
          Vector * Ba_, Vector * Cy_, double & Ca_, int option_=1) :
  B_prec(B_prec_), BT_prec(BT_prec_), A(A_), C(C_), option(option_), Ba(Ba_), Cy(Cy_), Ca(Ca_)
  {
    N = A_->Height();

    Array<int> offsets_(3);
    offsets_[0] = 0;
    offsets_[1] = N;
    offsets_[2] = 2 * N;

    offsets = &offsets_;
  }

  // set action
  virtual void Mult(const Vector &k, Vector &y) const
  {
    Vector vec1, vec2;
    vec1.MakeRef(const_cast<Vector&>(k), 0, N);
    vec2.MakeRef(const_cast<Vector&>(k), N, N);

    Vector out1, out2;
    out1.MakeRef(const_cast<Vector&>(y), 0, N);
    out2.MakeRef(const_cast<Vector&>(y), N, N);

    if (option == 0) {
      // block diag
      B_prec->Mult(vec1, out1);
      BT_prec->Mult(vec2, out2);

    } else if (option == 1) {
      // upper triangular
      // block GS
      Vector temp1(N), temp2(N);
      BT_prec->Mult(vec2, out2);

      // solve B out1 + A out2 = vec1
      A->Mult(out2, temp2);
      add(-1.0, temp2, 1.0, vec1, temp2);
      B_prec->Mult(temp2, out1);

    } else if (option == 2) {
      // lower triangular
      // block GS
      Vector temp1(N), temp2(N);
      B_prec->Mult(vec1, out1);

      // solve BT out2 + C out1 = vec2
      C->Mult(out1, temp1);
      add(-1.0, temp1, 1.0, vec2, temp1);
      BT_prec->Mult(temp1, out2);
      
    } else if (option == 3) {
      // block diag with woodbury
      Vector temp1(N), temp2(N);
      Vector temp3(N), temp4(N);
      double num, denom;

      // BMat
      // BT - 1 / Ca Cy BaT
      B_prec->Mult(vec1, temp1);
      B_prec->Mult(*Cy, temp2);
      num = - ((*Ba) * temp1) / Ca;
      denom = 1 - ((*Ba) * temp2) / Ca;
      add(1.0, temp1, - num / denom, temp2, out1);

      // BTMat
      // B - 1 / Ca Ba CyT
      BT_prec->Mult(vec2, temp3);
      BT_prec->Mult(*Ba, temp4);
      num = - ((*Cy) * temp3) / Ca;
      denom = 1 - ((*Cy) * temp4) / Ca;
      add(1.0, temp3, - num / denom, temp4, out2);
      
    
    } else if (option == 4) {
      // gauss seidel
      // block GS
      Vector temp1(N), temp2(N);
      int max_itr = 1;

      out1 = 0.0;
      for (int i = 0; i < max_itr; ++i) {
        // solve BT out2  = - C out1_old + vec2
        C->Mult(out1, temp1);
        add(-1.0, temp1, 1.0, vec2, temp1);
        BT_prec->Mult(temp1, out2);

        // solve B out1 + A out2 = vec1
        A->Mult(out2, temp2);
        add(-1.0, temp2, 1.0, vec1, temp2);
        B_prec->Mult(temp2, out1);
      }
    } else if (false) {
      Vector temp1(N), temp2(N);
      Vector temp3(N), temp4(N);
        
      // Schur decomp

      // [I  -A AMG(BT)] [vec1] = [temp1]
      // [0           I] [vec2] = [vec2]
      BT_prec->Mult(vec2, temp1);
      A->Mult(temp1, temp1);
      add(1.0, vec1, -1.0, temp1, temp1);

      // [AMG(B)        ] [temp1] = [out1]
      // [       AMG(BT)] [vec2]  = [temp2]
      B_prec->Mult(temp1, out1);
      BT_prec->Mult(vec2, temp2);

      // [I            0] [out1]  = [out1]
      // [- AMG(BT) C  I] [temp2] = [out2]
      C->Mult(out1, temp3);
      BT_prec->Mult(temp3, temp3);
      add(1.0, temp2, -1.0, temp3, out2);


    }
    

    // k.GetSubVector(*dofs1, *vec1);
    // k.GetSubVector(*dofs2, *vec2);


    // y.SetSubVector(*dofs1, *vec1);
    // y.SetSubVector(*dofs2, *vec2);


    
    out1.SyncAliasMemory(y);
    out2.SyncAliasMemory(y);
    // y = k;
  }
  
  virtual void SetOperator(const Operator &op) {};
  virtual ~SchurPC() {}
};





#endif
