#ifndef BOUNDARY
#define BOUNDARY

#include "mfem.hpp"
#include "plasma_model.hpp"
#include "elliptic_integral.hpp"
using namespace mfem;
using namespace std;



double N_coefficient(const Vector &x, const double & rho_gamma, const double & mu);
double M_coefficient(const Vector &x, const Vector &y, const double & mu);



/*
  Coefficient denoted as N(x) (order = 1) or M(x) (order = 2) in notes
*/
class BoundaryCoefficient : public Coefficient
{
private:
  // radius of far field boundary
  double rho_gamma;
  PlasmaModel *model;
  int order;
  Vector y;
public:
  BoundaryCoefficient(double & rho_gamma_, PlasmaModel *model_, int order_) : rho_gamma(rho_gamma_), model(model_), order(order_) { }
  virtual void SetY(Vector & y_) {y=y_;}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~BoundaryCoefficient() { }
};

class DoubleIntegralCoefficient : public Coefficient
{
private:
  // radius of far field boundary
  BoundaryCoefficient *boundary_coeff;
  const GridFunction *psi;
  GridFunction *ones;
  FiniteElementSpace *fespace;
public:
  DoubleIntegralCoefficient(BoundaryCoefficient *boundary_coeff_, FiniteElementSpace *fespace_) : boundary_coeff(boundary_coeff_), fespace(fespace_) { ones = new GridFunction(fespace); ones[0] = 1.0;}
  void set_grid_function(const GridFunction *psi_) {psi = psi_;}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~DoubleIntegralCoefficient() { }
};


#endif
