#ifndef SYS_OPERATOR
#define SYS_OPERATOR

#include "mfem.hpp"
#include "plasma_model.hpp"
using namespace mfem;
using namespace std;

double GetMaxError(LinearForm &res);

/*
  Mult:
  diff_operator * psi - plasma_term(psi) * psi - coil_term

  GetGradient:
  diff_operator - sum_{i=1}^3 diff_plasma_term_i(psi)
 */
class SysOperator : public Operator
{
private:
  BilinearForm *diff_operator;
  LinearForm *coil_term;
  PlasmaModel *model;
  FiniteElementSpace *fespace;
  Mesh *mesh;
  map<int, vector<int>> vertex_map;
  mutable SparseMatrix *Mat;
  int attr_lim;
public:
  SysOperator(BilinearForm *diff_operator_, LinearForm *coil_term_,
              PlasmaModel *model_, FiniteElementSpace *fespace_,
              Mesh *mesh_, int attr_lim_) :
    diff_operator(diff_operator_), coil_term(coil_term_),
    model(model_), fespace(fespace_),
    mesh(mesh_), Mat(NULL), attr_lim(attr_lim_) {
    vertex_map = compute_vertex_map(*mesh, attr_lim);
  }
  virtual void Mult(const Vector &psi, Vector &y) const;
  virtual Operator &GetGradient(const Vector &psi) const;
  virtual ~SysOperator() { };
};



#endif
