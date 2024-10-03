#include "mfem.hpp"
#include "funs.hpp"
#include "linear_solver.hpp"


namespace mfem
{

class DesignDensity
{
private:
   FiniteElementSpace &fes_control;
   const real_t tot_vol;
   const real_t min_vol;
   const real_t max_vol;
   bool hasPassiveElements;
   LegendreEntropy *entropy;
   std::unique_ptr<GridFunction> zero;
public:
   DesignDensity(
      FiniteElementSpace &fes_control, const real_t tot_vol,
      const real_t min_vol, const real_t max_vol,
      LegendreEntropy *entropy=nullptr);

   real_t ApplyVolumeProjection(GridFunction &x);
};

class DensityBasedTopOpt
{
private:
   FunctionCoefficient simp_cf;
   FunctionCoefficient der_simp_cf;
   ProductCoefficient simp_lamba_cf;
   ProductCoefficient simp_mu_cf;
   bool isCompliance;
public:
   DensityBasedTopOpt(const real_t lambda, const real_t mu, const real_t rho0,
                      const real_t exponent, const real_t min_vol, const real_t max_vol,
                      LegendreEntropy *entropy, bool isCompliance)
   :simp_cf([exponent, rho0](const real_t x){return simp(x, exponent, rho0);}), der_simp_cf([exponent, rho0](const real_t x){return der_simp(x, exponent, rho0);})
   {
   }

}

} // end of namespace mfem
