// =============================================================================
// Objective Functional for Transient Topology Optimization
// =============================================================================
//
// Minimize displacement in protected subdomain:
//   J = ∫₀ᵀ ∫_Ω̃ |u|² dx dt
//
// Based on paper Section 5.1 (eq. 840-844)
// =============================================================================

#ifndef OBJECTIVE_FUNCTIONAL_HPP
#define OBJECTIVE_FUNCTIONAL_HPP

#include "mfem.hpp"

namespace mfem
{

// =============================================================================
// Subdomain Indicator Coefficient
// =============================================================================
// χ_Ω̃(x) = 1 if x ∈ Ω̃, 0 otherwise
// For circular subdomain: |x - x_c| < r
class SubdomainIndicator : public Coefficient
{
private:
   real_t x_center, y_center, radius;

public:
   SubdomainIndicator(real_t xc, real_t yc, real_t r)
      : x_center(xc), y_center(yc), radius(r) {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);

      real_t dx = x(0) - x_center;
      real_t dy = x(1) - y_center;
      real_t dist = sqrt(dx*dx + dy*dy);

      return (dist < radius) ? 1.0 : 0.0;
   }
};

// =============================================================================
// Time-Integrated Objective Functional
// =============================================================================
// J = ∫₀ᵀ ∫_Ω̃ |u(t)|² dx dt
//
// Discrete form (with trapezoidal rule in time):
// J_h = Σₙ ωₙ ∫_Ω̃ |u^n|² dx
//
class TimeIntegratedObjective
{
private:
   ParFiniteElementSpace *fespace;
   Coefficient *subdomain_indicator;

   real_t accumulated_objective;
   Array<real_t> quadrature_weights;  // Time quadrature: ωₙ

   MPI_Comm comm;
   int myid;

public:
   TimeIntegratedObjective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_)
      : fespace(fes), subdomain_indicator(indicator),
        accumulated_objective(0.0), comm(comm_)
   {
      MPI_Comm_rank(comm, &myid);
   }

   // Initialize for new optimization iteration
   void Reset() { accumulated_objective = 0.0; }

   // Accumulate contribution at timestep n
   // u: displacement at time t^n
   // dt: timestep size
   // quadrature_type: "trapezoidal", "midpoint", "simpson"
   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                              int timestep, int total_steps)
   {
      // Compute ∫_Ω̃ |u|² dx using subdomain indicator
      ParLinearForm integrand(fespace);

      // |u|² coefficient
      GridFunctionCoefficient u_coef(const_cast<ParGridFunction*>(&u));

      // Custom coefficient: χ_Ω̃(x) * |u(x)|²
      class WeightedNormSquared : public Coefficient
      {
      private:
         GridFunctionCoefficient *u_cf;
         Coefficient *chi;
      public:
         WeightedNormSquared(GridFunctionCoefficient *uc, Coefficient *c)
            : u_cf(uc), chi(c) {}

         virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
         {
            Vector u_val;
            u_cf->Eval(u_val, T, ip);
            real_t u_norm_sq = u_val * u_val;
            real_t chi_val = chi->Eval(T, ip);
            return chi_val * u_norm_sq;
         }
      };

      WeightedNormSquared weighted_u(&u_coef, subdomain_indicator);
      integrand.AddDomainIntegrator(new DomainLFIntegrator(weighted_u));
      integrand.Assemble();

      // Sum across processors
      real_t local_integral = integrand.Sum();
      real_t global_integral;
      MPI_Allreduce(&local_integral, &global_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      // Time quadrature weight (trapezoidal rule)
      real_t omega;
      if (timestep == 0 || timestep == total_steps - 1)
      {
         omega = 0.5 * dt;  // Half weight at endpoints
      }
      else
      {
         omega = dt;  // Full weight for interior points
      }

      real_t contribution = omega * global_integral;
      accumulated_objective += contribution;

      return contribution;
   }

   // Get total accumulated objective
   real_t GetObjective() const { return accumulated_objective; }

   // Compute objective gradient at timestep n: ∂J_Ω/∂u
   // This becomes the RHS for the adjoint equation
   //
   // From paper eq. 877-882:
   //   ∂J_Ω/∂u = 2 χ_Ω̃(x) u(x)
   //
   void ComputeObjectiveGradient(const ParGridFunction &u,
                                  real_t dt, int timestep, int total_steps,
                                  ParLinearForm &grad_form)
   {
      // Time quadrature weight
      real_t omega;
      if (timestep == 0 || timestep == total_steps - 1)
         omega = 0.5 * dt;
      else
         omega = dt;

      // ∂J_Ω/∂u = 2 ω χ_Ω̃(x) u(x)
      GridFunctionCoefficient u_coef(const_cast<ParGridFunction*>(&u));

      class ObjectiveGradientCoef : public VectorCoefficient
      {
      private:
         GridFunctionCoefficient *u_cf;
         Coefficient *chi;
         real_t weight;
      public:
         ObjectiveGradientCoef(int dim, GridFunctionCoefficient *uc,
                               Coefficient *c, real_t w)
            : VectorCoefficient(dim), u_cf(uc), chi(c), weight(w) {}

         void Eval(Vector &V, ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            u_cf->Eval(V, T, ip);  // Get u(x)
            real_t chi_val = chi->Eval(T, ip);
            V *= 2.0 * weight * chi_val;  // 2 ω χ_Ω̃ u
         }
      };

      int dim = u.FESpace()->GetMesh()->Dimension();
      ObjectiveGradientCoef grad_coef(dim, &u_coef, subdomain_indicator, omega);

      grad_form.AddDomainIntegrator(new VectorDomainLFIntegrator(grad_coef));
      grad_form.Assemble();
   }
};

} // namespace mfem

#endif // OBJECTIVE_FUNCTIONAL_HPP
