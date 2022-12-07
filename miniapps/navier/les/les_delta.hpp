#ifndef MFEM_NAVIER_LES_DELTA
#define MFEM_NAVIER_LES_DELTA

#include <mfem.hpp>

namespace mfem
{
namespace navier
{

/**
 * @brief Abstact base class for delta value computation for LES subgrid scale
 * models on each quadrature point.
 */
class LESDelta : public Coefficient {};

/**
 * @brief A geometric delta (cell "width") computation that is valid for curved
 * elements.
 */
class CurvedGeometricDelta : public LESDelta
{
public:
   CurvedGeometricDelta(const int order) :
      order(order) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return T.Jacobian().CalcSingularvalue(T.GetDimension()-1) /
             (order + 1);
   };

private:
   const int order;
};

/**
 * @brief Van Driest damping applied to a geometric delta.
 */
class VanDriestDelta : public LESDelta
{
   using yplus_f_type = std::function<double(const Vector& u, const double wd)>;

public:
   /**
    * @brief Construct a new Van Driest Delta object
    *
    * The default to compute yplus is the 1/7th rule.
    *
    * @param[in] geometric_delta geometric delta function to be called
    * @param[in] velocity velocity GridFunction
    * @param[in] wall_distance wall distance GridFunction
    * @param[in] Re Reynolds number
    * @param[in] kappa parameter
    * @param[in] aplus parameter
    * @param[in] cdelta parameter
    * @param[in] yplus_f user supplied yplus function callback
    */
   VanDriestDelta(LESDelta &geometric_delta,
                  const GridFunction &velocity,
                  const GridFunction &wall_distance,
                  const double Re,
                  const double kappa = 0.41,
                  const double aplus = 26.0,
                  const double cdelta = 0.158,
                  yplus_f_type yplus_f = {}) :
      geometric_delta(geometric_delta),
      velocity(velocity),
      wall_distance(wall_distance),
      Re(Re),
      cf(0.026/std::pow(Re, 1.0/7.0)),
      kappa(kappa),
      aplus(aplus),
      cdelta(cdelta),
      yplus_f(yplus_f)
   {
      if (!this->yplus_f)
      {
         this->yplus_f = [&](const Vector& u, const double wd)
         {
            // Compute yplus by the 1/7th rule
            const double tau_w = 0.5 * 1.0 * u.Norml2() * u.Norml2() * cf;
            const double u_tau = sqrt(tau_w);
            const double yplus = wd * u_tau * Re;
            return yplus;
         };
      }
   }

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const double geometric_delta_val = geometric_delta.Eval(T, ip);
      const double wd = wall_distance.GetValue(T, ip);
      velocity.GetVectorValue(T, ip, u);
      const double yplus = yplus_f(u, wd);
      const double D = 1.0 - exp(-yplus / aplus);
      return std::min(kappa/cdelta * wd * D, geometric_delta_val);
   }

private:
   /// Geometric delta. Used as "backup".
   LESDelta &geometric_delta;

   /// Velocity
   const GridFunction &velocity;

   /// Wall distance
   const GridFunction &wall_distance;

   /// Reynolds number
   const double Re;

   /// Parameter
   const double cf, kappa, aplus, cdelta;

   /// Storage for velocity on
   Vector u;

   /// Callback to compute yplus on each quadrature point
   yplus_f_type yplus_f;
};

}
}

#endif