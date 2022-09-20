#ifndef DIFFUSION_TERM
#define DIFFUSION_TERM

#include "mfem.hpp"
#include "plasma_model.hpp"
using namespace mfem;
using namespace std;

double one_over_r_mu(const Vector & x, double & mu);

/*
  Coefficient for diffusion integrator.
*/
class DiffusionIntegratorCoefficient : public Coefficient
{
private:
  PlasmaModel *model;
public:
  DiffusionIntegratorCoefficient(PlasmaModel *model_) : model(model_) { }
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~DiffusionIntegratorCoefficient() { }
};


#endif
