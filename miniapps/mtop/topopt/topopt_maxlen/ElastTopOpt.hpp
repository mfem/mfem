#include "mfem.hpp"
#include <cmath>
#include <memory>

using namespace std;
using namespace mfem;

// SIMP coefficient: r(rho) = E_min + rho^exponent (E_max - E_min).
// The argument is either a grid function directly or any coefficient, so SIMP
// can be composed with a projection, e.g. r(H(rho~)).
class SIMPCoefficient : public Coefficient
{
protected:
   GridFunctionCoefficient rho_gf_cf;
   Coefficient *rho_cf;
   real_t E_min, E_max, exponent;

public:
   SIMPCoefficient(GridFunction *rho_, real_t E_min_ = 1e-6,
                   real_t E_max_ = 1.0, real_t exponent_ = 3.0)
      : rho_gf_cf(rho_), rho_cf(&rho_gf_cf), E_min(E_min_), E_max(E_max_),
        exponent(exponent_) { }

   SIMPCoefficient(Coefficient &rho_, real_t E_min_ = 1e-6,
                   real_t E_max_ = 1.0, real_t exponent_ = 3.0)
      : rho_cf(&rho_), E_min(E_min_), E_max(E_max_), exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_cf->Eval(T, ip);
      val = std::max(0.0, std::min(1.0, val));
      // r(rho) = E_min + rho^exponent (E_max - E_min)
      return E_min + std::pow(val, exponent) * (E_max - E_min);
   }
};

class SIMPGradCoefficient : public Coefficient
{
protected:
   GridFunctionCoefficient rho_gf_cf;
   Coefficient *rho_cf;
   real_t E_min, E_max, exponent;

public:
   SIMPGradCoefficient(GridFunction *rho_, real_t E_min_ = 1e-6,
                   real_t E_max_ = 1.0, real_t exponent_ = 3.0)
      : rho_gf_cf(rho_), rho_cf(&rho_gf_cf), E_min(E_min_), E_max(E_max_),
        exponent(exponent_) { }

   SIMPGradCoefficient(Coefficient &rho_, real_t E_min_ = 1e-6,
                   real_t E_max_ = 1.0, real_t exponent_ = 3.0)
      : rho_cf(&rho_), E_min(E_min_), E_max(E_max_), exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_cf->Eval(T, ip);
      val = std::max(0.0, std::min(1.0, val));
      // r'(rho) = exponent * rho^(exponent-1) (E_max - E_min)
      return exponent * std::pow(val, exponent - 1.0) * (E_max - E_min);
   }
};

class HeavisideCoefficient : public Coefficient
{
protected:
   GridFunctionCoefficient rho_filter_cf;
   real_t beta, eta;

public:
   HeavisideCoefficient(GridFunction *rho_filter_, real_t beta_ = 1.0, real_t eta_ = 0.5)
      : rho_filter_cf(rho_filter_), beta(beta_), eta(eta_) { }

   void SetBeta(real_t beta_) { beta = beta_; }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter_cf.Eval(T, ip);
      val = std::max(0.0, std::min(1.0, val));
      // H(rho~) = (tanh(beta*eta) + tanh(beta*(rho~-eta))) / (tanh(beta*eta) + tanh(beta*(1 - eta)))
      real_t num = std::tanh(beta * eta) + std::tanh(beta * (val - eta));
      real_t den = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
      return num / den;
   }
};

class HeavisideGradCoefficient : public Coefficient
{
protected:
   GridFunctionCoefficient rho_filter_cf;
   real_t beta, eta;

public:
   HeavisideGradCoefficient(GridFunction *rho_filter_, real_t beta_ = 1.0, real_t eta_ = 0.5)
      : rho_filter_cf(rho_filter_), beta(beta_), eta(eta_) { }

   void SetBeta(real_t beta_) { beta = beta_; }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter_cf.Eval(T, ip);
      val = std::max(0.0, std::min(1.0, val));
      // H'(rho~) = beta * (1 - tanh^2(beta*(rho~-eta))) / (tanh(beta*eta) + tanh(beta*(1 - eta)))
      real_t num = beta * (1.0 - std::pow(std::tanh(beta * (val - eta)), 2));
      real_t den = std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta));
      return num / den;
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