#ifndef SYS_OPERATOR
#define SYS_OPERATOR

#include "mfem.hpp"
#include "plasma_model.hpp"
using namespace mfem;
using namespace std;

double GetMaxError(LinearForm &res);
double GetMaxError(GridFunction &res);

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
  mutable PlasmaModel *model;
  FiniteElementSpace *fespace;
  Mesh *mesh;
  map<int, vector<int>> vertex_map;
  mutable SparseMatrix *Mat;
  int attr_lim;
  Array<int> *boundary_dofs;
  GridFunction *u_boundary;
public:
  SysOperator(BilinearForm *diff_operator_, LinearForm *coil_term_,
              PlasmaModel *model_, FiniteElementSpace *fespace_,
              Mesh *mesh_, int attr_lim_, Array<int> *boundary_dofs_,
              GridFunction *u_boundary_) :
    Operator(fespace_->GetTrueVSize()), diff_operator(diff_operator_), coil_term(coil_term_),
    model(model_), fespace(fespace_),
    mesh(mesh_), Mat(NULL), attr_lim(attr_lim_), boundary_dofs(boundary_dofs_),
    u_boundary(u_boundary_) {
    vertex_map = compute_vertex_map(*mesh, attr_lim);
  }
  virtual void Mult(const Vector &psi, Vector &y) const;
  virtual Operator &GetGradient(const Vector &psi) const;
  virtual ~SysOperator() { };
};



#endif
