#ifndef FIELD
#define FIELD

#include "mfem.hpp"
#include "plasma_model.hpp"

using namespace mfem;
using namespace std;

class FieldCoefficient : public Coefficient
{
private:
  GridFunction * psi;
  GridFunction * psi_z;
  GridFunction * psi_r;
  
  PlasmaModelBase * model;
  FiniteElementSpace fespace;
  double psi_ma;
  double psi_x;
  int comp;
public:
  FieldCoefficient(GridFunction * psi_,
                   GridFunction * psi_r_, GridFunction * psi_z_,
                   PlasmaModelBase *pm,
                   FiniteElementSpace & fespace_, int comp_) :
    psi(psi_), psi_r(psi_r_), psi_z(psi_z_),
    model(pm), fespace(fespace_), comp(comp_) { }

  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~FieldCoefficient() { }

  void set_psi_vals(double & psi_x_, double & psi_ma_) {
    psi_ma = psi_ma_;
    psi_x = psi_x_;
  }
};


#endif
