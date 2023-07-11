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
  SparseMatrix *F;
  SparseMatrix *H;
  SparseMatrix *K;
  vector<Vector> *alpha_coeffs;
  vector<Array<int>> *J_inds;

  bool include_plasma = true;

  Vector *g;
  Vector *uv_currents;
  mutable Vector Plasma_Vec;
  mutable double Alpha_Term;
  mutable double Plasma_Current;
  mutable Vector B_alpha;

  mutable double psi_x;
  mutable double psi_ma;

  mutable double * x_x;
  mutable double * x_ma;

  mutable int ind_ma;
  mutable int ind_x;
  mutable set<int> plasma_inds;
  
  double *alpha_bar;
  int N_control;
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
    N_control = alpha_coeffs->size();
    
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

    
  }
  virtual void Mult(const Vector &psi, Vector &y) const;
  virtual Operator &GetGradient(const Vector &psi) const;
  virtual ~SysOperator() { };

  void compute_psi_N_derivs(const Vector &psi, int k, double * psi_N,
                            Vector * grad_psi_N, SparseMatrix * hess_psi_N);

  double compute_obj(const Vector &psi);
  Vector* compute_grad_obj(const Vector &psi);
  SparseMatrix* compute_hess_obj(const Vector &psi);
  
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
  double* get_x_x() {
    return x_x;
  }
  double* get_x_ma() {
    return x_ma;
  }
    
};



#endif
