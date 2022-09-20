#ifndef MFEM_NAVIER_SMAGORINSKY
#define MFEM_NAVIER_SMAGORINSKY

#include <mfem.hpp>
#include "lib/turbulence_model.hpp"
#include "les/les_delta.hpp"

namespace mfem
{
namespace navier
{

/**
 * @brief Smagorinsky subgrid scale turbulence model for LES computations.
 *
 * Smagorinsky, J. (1963)
 *
 * The subgrid scale stress tensor is defined as
 *  B = 2/3*k*I - 2*nu_t*dev(S)
 * where
 *  S = symmetric_grad(u)
 *  nu_t = Ck*sqrt(k)*delta
 *  S:B + Ce*k^3/2 / delta = 0
 *
 * Carrying out the double dot product we arrive at
 *  k = (-b+sqrt(b^2+4ac)/(2a))^2
 * where
 *  a = Ce/delta
 *  b = 2/3 tr(S)
 *  c = 2 Ck delta (dev(S) : S)
 * For incompressible flow we make use of the following simplifications
 *  b = 0, tr(S) = 0 for div(u) = 0
 *  c = Ck delta |S|^2, with |S| = sqrt(2 S:S)
 * from that we get
 *  k = c / a
 *  nu_t = Ck sqrt(Ck/Ce) delta^2 |S|
 */
class Smagorinsky : public TurbulenceModel
{
private:
   class SmagorinskyCoefficient : public Coefficient
   {
   public:
      SmagorinskyCoefficient(const double ck,
                             const double ce,
                             const GridFunction &vgf,
                             LESDelta &delta) :
         p(vgf.FESpace()->GetOrder(0)),
         ck(ck),
         ce(ce),
         vgf(vgf),
         delta(delta) {}

      double Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override
      {
         const double delta_val = delta.Eval(T, ip);
         double abs_S = 0.0;
         vgf.GetVectorGradient(T, S);

         // S = 0.5 * (\nabla v + \nabla^T v)
         S.Symmetrize();

         for (int i = 0; i < S.NumRows(); i++)
         {
            for (int j = 0; j < S.NumCols(); j++)
            {
               abs_S += S(i,j) * S(i,j);
            }
         }
         abs_S = sqrt(2.0 * abs_S);

         return ck * sqrt(ck/ce) * delta_val * delta_val * abs_S;
      }

   private:
      /// Polynomial order of the solution space
      const int p;
      const double ck, ce;
      const GridFunction &vgf;
      DenseMatrix S;
      LESDelta &delta;
   };

public:
   /**
    * @brief Construct a new Smagorinsky object
    *
    * @param[in] velocity velocity GridFunction
    * @param[in] delta geometric delta
    * @param[in] ck parameter
    * @param[in] ce parameter
    */
   Smagorinsky(const GridFunction &velocity,
               LESDelta &delta,
               const double ck = 0.094,
               const double ce = 1.048) :
      velocity(velocity),
      delta(delta),
      ck(ck),
      ce(ce),
      coeff(ck, ce, velocity, delta) {}

   /**
    * @brief Compute the eddy viscosity
    *
    * @param[in,out] nu eddy/turbulent viscosity
    */
   void ComputeEddyViscosity(ParGridFunction &nu) override
   {
      nu.ProjectCoefficient(coeff);
   }

private:
   /// Velocity
   const GridFunction &velocity;

   /// Parameter
   const double ck, ce;

   /// Coefficient representation of the Smagorinsky SGS model
   SmagorinskyCoefficient coeff;

   /// Length scale computation
   LESDelta &delta;
};
}
}

#endif