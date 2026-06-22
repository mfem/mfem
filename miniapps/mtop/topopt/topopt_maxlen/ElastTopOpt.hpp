#include "mfem.hpp"
#include <cmath>
#include "QuantityOfInterest.hpp"

using namespace std;
using namespace mfem;


// SIMP coefficient: r(rho~) = E_min + rho~^exponent (E_max - E_min).
class SIMPCoefficient : public Coefficient
{
protected:
   GridFunction *rho_filter;
   real_t E_min, E_max, exponent;

public:
   SIMPCoefficient(GridFunction *rho_filter_, real_t E_min_ = 1e-6,
                   real_t E_max_ = 1.0, real_t exponent_ = 3.0)
      : rho_filter(rho_filter_), E_min(E_min_), E_max(E_max_),
        exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter->GetValue(T, ip);
      // r(rho~) = E_min + rho~^exponent (E_max - E_min)
      return E_min + std::pow(val, exponent) * (E_max - E_min);
   }
};

class SIMPGradCoefficient : public SIMPCoefficient
{
public:
   using SIMPCoefficient::SIMPCoefficient;   // inherits members + constructor

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter->GetValue(T, ip);
      // r'(rho~) = exponent * rho~^(exponent-1) (E_max - E_min)
      return exponent * std::pow(val, exponent - 1.0) * (E_max - E_min);
   }
};

// Strain-energy-density (compliance sensitivity) coefficient.
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient *lambda = nullptr;
   Coefficient *mu = nullptr;
   GridFunction *u = nullptr;             // displacement state
   DenseMatrix grad;

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_, 
                                  GridFunction *u_)
      : lambda(lambda_), mu(mu_), u(u_)
   {
      MFEM_ASSERT(u, "displacement not set");
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t L = lambda->Eval(T, ip);
      real_t M = mu->Eval(T, ip);
      u->GetVectorGradient(T, grad);
      real_t div_u = grad.Trace();

      // psi0(u) = lambda (div u)^2 + 2 mu |eps(u)|^2  (strain energy density)
      real_t density = L * div_u * div_u;
      int dim = T.GetSpaceDim();
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            density += M * grad(i, j) * (grad(i, j) + grad(j, i));
         }
      }
      return density;
   }
};
