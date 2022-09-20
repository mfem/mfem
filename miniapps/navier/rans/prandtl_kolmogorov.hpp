#ifndef MFEM_NAVIER_PRANDTL_KOLMOGOROV_HPP
#define MFEM_NAVIER_PRANDTL_KOLMOGOROV_HPP

#include <mfem.hpp>
#include "scalar.hpp"
#include "turbulence_model.hpp"

namespace mfem
{
namespace navier
{

/**
 * @brief The one equation RANS turbulence model as described by Prandtl and
 * Kolmogorov.
 */
class PrandtlKolmogorov : public ScalarEquation, public TurbulenceModel
{
public:
   /**
    * @brief Compute the sqrt of the positive part of k
    *
    * @param[in] k turbulent kinetic energy
    * @return sqrt of the positive part of k
    */
   static inline
   double sqrtkpos(const double k)
   {
      return sqrt(0.5*(k+abs(k)));
   }

   /**
    * @brief Compute the turbulent length scale
    *
    * @param[in] k turbulent kinetic energy
    * @param[in] wd wall distance
    * @return turbulent length scale
    */
   static inline
   double tls(const double k, const double wd)
   {
      const double L = 0.3, tau = 1.0;

      double tls = std::min(sqrt(2.0)*sqrtkpos(k)*tau, 0.41*wd*sqrt(wd/L));
      if (tls == 0.0)
      {
         tls = 1e-8;
      }
      return tls;
   }

   /**
    * @brief Compute the eddy viscosity
    *
    * @param k turbulent kinetic energy
    * @param tls turbulent length scale
    * @param mu parameter
    * @return eddy viscosity
    */
   static inline
   double eddy_viscosity(double k, double tls,
                         double mu)
   {
      return mu * tls * sqrtkpos(k);
   }

   /**
    * @brief Coefficient which represents the eddy viscosity
    */
   class EddyViscosityCoefficient : public Coefficient
   {
   public:
      EddyViscosityCoefficient(const ParGridFunction &kgf,
                               const ParGridFunction &wdgf,
                               const double mu) :
         kgf(kgf),
         wdgf(wdgf),
         mu(mu) {}

      double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
      {
         const double k = kgf.GetValue(T, ip);
         const double wd = wdgf.GetValue(T, ip);

         return eddy_viscosity(k, tls(k, wd), mu);
      }

   private:
      /// Turbulent kinetic energy
      const ParGridFunction &kgf;

      /// Wall distance
      const ParGridFunction &wdgf;

      /// Parameter
      const double mu;
   };

   /**
    * @brief Coefficient that represents the reaction type part of the residual
    */
   class ReactionCoefficient : public Coefficient
   {
   public:
      ReactionCoefficient(const GridFunction &kgf,
                          const GridFunction &vgf,
                          const GridFunction &wdgf,
                          double mu) :
         kgf(kgf), vgf(vgf), wdgf(wdgf), mu(mu) {}

      double Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override
      {
         const double k = kgf.GetValue(T, ip);
         const double wd = wdgf.GetValue(T, ip);

         vgf.GetVectorGradient(T, S);

         // S = 0.5 * (\nabla v + \nabla^T v)
         S.Symmetrize();

         double abs_S_squared = 0.0;
         for (int i = 0; i < S.NumRows(); i++)
         {
            for (int j = 0; j < S.NumCols(); j++)
            {
               abs_S_squared += S(i,j) * S(i,j);
            }
         }

         const double tlsval = tls(k, wd);

         return -1.0 / tlsval * k * sqrtkpos(k) + eddy_viscosity(k, tlsval,
                                                                 mu) * abs_S_squared;
      }

   private:
      /// Turbulent kinetic energy
      const GridFunction &kgf;

      /// Velocity
      const GridFunction &vgf;

      /// Wall distance
      const GridFunction &wdgf;

      /// Parameter
      const double mu;

      /// Symmetric gradient of the velocity
      DenseMatrix S;
   };

   /**
    * @brief Coefficient which represents the spatially varying viscosity in the
    * equation
    */
   class ViscosityCoefficient : public Coefficient
   {
   public:
      ViscosityCoefficient(const GridFunction &nugf, const GridFunction &kgf,
                           const GridFunction &wdgf, const double mu) :
         nugf(nugf), kgf(kgf), wdgf(wdgf), mu(mu) {}

      double Eval(ElementTransformation &T,
                  const IntegrationPoint &ip) override
      {
         const double k = kgf.GetValue(T, ip);
         const double nu = nugf.GetValue(T, ip);
         const double wd = wdgf.GetValue(T, ip);

         const double tlsval = tls(k, wd);

         return nu + eddy_viscosity(k, tlsval, mu);
      }

   private:
      /// Spatially varying viscosity
      const GridFunction &nugf;

      /// Turbulent kinetic energy
      const GridFunction &kgf;

      /// Wall distance
      const GridFunction &wdgf;

      /// Parameter
      const double mu;
   };

   PrandtlKolmogorov(ParMesh &mesh, const int order,
                     ParGridFunction &velgf,
                     ParGridFunction &nugf,
                     ParGridFunction &wdgf);

   /**
    * @brief Compute the turbulent/eddy viscosity
    *
    * Method which computes the eddy viscosity at the current state and time of
    * the k-equation. It is necessary to keep the k-equation at the same
    * timestep as the Navier-Stokes momentum equation. In this implementation a
    * coefficient is simply projected onto the ParGridFunction.
    *
    * @param[in,out] nu eddy viscosity
    */
   void ComputeEddyViscosity(ParGridFunction &nu) override;

protected:
   /// Velocity
   ParGridFunction &velgf;

   /// Spatially varying viscosity
   ParGridFunction &nugf;

   /// Wall distance
   ParGridFunction &wdgf;

   /// Parameter
   const double mu = 0.55;

   ViscosityCoefficient viscosity_coeff;
   ReactionCoefficient reaction_coeff;
   EddyViscosityCoefficient eddy_viscosity_coeff;
};

}
}

#endif