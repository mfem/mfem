// =============================================================================
// Objective Functionals for Transient Topology Optimization
// =============================================================================
//
// This header defines:
//   - TimeIntegratedObjective: abstract base class for objectives
//   - DisplacementL2Objective: minimize |u(t)|² (wave shielding)
//   - (Add more objectives here as subclasses)
//
// INTERFACE:
//   - AccumulateTimestep(u, dt, step, total_steps) → returns contribution
//   - ComputeObjectiveGradient(u, dt, step, total_steps, grad_form) → fills ∂J/∂u
//
// USAGE:
//   TimeIntegratedObjective *obj = new DisplacementL2Objective(...);
//   // Solver calls obj->AccumulateTimestep() in forward sweep
//   // Solver calls obj->ComputeObjectiveGradient() in adjoint sweep
//
// =============================================================================

#ifndef OBJECTIVE_FUNCTIONAL_HPP
#define OBJECTIVE_FUNCTIONAL_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <utility>

namespace mfem
{

class SubdomainIndicator : public Coefficient
{
private:
   real_t x_center, y_center, radius;

public:
   SubdomainIndicator(real_t xc, real_t yc, real_t r)
      : x_center(xc), y_center(yc), radius(r) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);

      const real_t dx = x(0) - x_center;
      const real_t dy = x(1) - y_center;
      const real_t dist = std::sqrt(dx*dx + dy*dy);

      return (dist < radius) ? 1.0 : 0.0;
   }
};

// =============================================================================
// ABSTRACT BASE CLASS: TimeIntegratedObjective
// =============================================================================
// Interface for time-integrated objective functionals J = ∫_0^T ∫_Ω j(u,t) dx dt
//
// Subclasses must implement:
//   - AccumulateTimestep: compute contribution at one timestep
//   - ComputeObjectiveGradient: compute ∂J/∂u at one timestep (for adjoint)
//
class TimeIntegratedObjective
{
protected:
   ParFiniteElementSpace *fespace;
   real_t accumulated_objective;
   MPI_Comm comm;
   int myid;

   // Trapezoidal rule weights for time integration
   real_t TimeWeight(real_t dt, int timestep, int total_steps) const
   {
      return (timestep == 0 || timestep == total_steps - 1) ? 0.5 * dt : dt;
   }

public:
   TimeIntegratedObjective(ParFiniteElementSpace *fes, MPI_Comm comm_)
      : fespace(fes), accumulated_objective(0.0), comm(comm_)
   {
      MPI_Comm_rank(comm, &myid);
   }

   virtual ~TimeIntegratedObjective() = default;

   void Reset() { accumulated_objective = 0.0; }
   real_t GetObjective() const { return accumulated_objective; }

   /// Accumulate objective contribution at current timestep
   /// @return contribution added (for monitoring)
   virtual real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                                      int timestep, int total_steps) = 0;

   /// Compute objective gradient ∂J/∂u at current timestep (for adjoint)
   virtual void ComputeObjectiveGradient(const ParGridFunction &u,
                                         real_t dt, int timestep, int total_steps,
                                         ParLinearForm &grad_form) = 0;
};

// =============================================================================
// DISPLACEMENT L2 OBJECTIVE: minimize ∫∫ |u(t)|² dx dt in subdomain
// =============================================================================
class DisplacementL2Objective : public TimeIntegratedObjective
{
private:
   Coefficient *subdomain_indicator; // non-owning view used in hot paths
   std::unique_ptr<Coefficient> owned_indicator;

public:
   /// Borrow an externally-owned indicator coefficient.
   DisplacementL2Objective(ParFiniteElementSpace *fes,
                           Coefficient &indicator,
                           MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        subdomain_indicator(&indicator) {}

   /// Take ownership of an indicator coefficient.
   DisplacementL2Objective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator,
                           MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        subdomain_indicator(indicator.get()),
        owned_indicator(std::move(indicator)) {}

   /// Backward-compatible constructor for legacy call sites.
   DisplacementL2Objective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : TimeIntegratedObjective(fes, comm_),
        subdomain_indicator(indicator),
        owned_indicator(own_indicator ? indicator : nullptr) {}

   virtual ~DisplacementL2Objective() = default;

   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                             int timestep, int total_steps) override
   {
      real_t local_integral = 0.0;
      Vector u_val;

      for (int e = 0; e < fespace->GetNE(); e++)
      {
         const FiniteElement *el = fespace->GetFE(e);
         ElementTransformation *T = fespace->GetElementTransformation(e);
         const int int_order = 2 * el->GetOrder() + 2;
         const IntegrationRule &ir = IntRules.Get(el->GetGeomType(), int_order);

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            u.GetVectorValue(*T, ip, u_val);

            const real_t u_norm_sq = u_val * u_val;
            const real_t chi_val = subdomain_indicator->Eval(*T, ip);
            local_integral += ip.weight * T->Weight() * chi_val * u_norm_sq;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&local_integral, &global_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      const real_t contribution = TimeWeight(dt, timestep, total_steps)
                                  * global_integral;
      accumulated_objective += contribution;

      return contribution;
   }

   void ComputeObjectiveGradient(const ParGridFunction &u,
                                 real_t dt, int timestep, int total_steps,
                                 ParLinearForm &grad_form) override
   {
      const real_t omega = TimeWeight(dt, timestep, total_steps);
      VectorGridFunctionCoefficient u_coef(&u);

      class ObjectiveGradientCoef : public VectorCoefficient
      {
      private:
         VectorGridFunctionCoefficient *u_cf;
         Coefficient *chi;
         real_t weight;

      public:
         ObjectiveGradientCoef(int vdim, VectorGridFunctionCoefficient *uc,
                               Coefficient *c, real_t w)
            : VectorCoefficient(vdim), u_cf(uc), chi(c), weight(w) {}

         void Eval(Vector &V, ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            u_cf->Eval(V, T, ip);
            const real_t chi_val = chi->Eval(T, ip);
            V *= 2.0 * weight * chi_val;
         }
      };

      ObjectiveGradientCoef grad_coef(u.VectorDim(), &u_coef,
                                      subdomain_indicator, omega);

      class HighOrderVectorDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape, q_vec;
         VectorCoefficient &q;

      public:
         HighOrderVectorDomainLFIntegrator(VectorCoefficient &q_)
            : q(q_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int vdim = q.GetVDim();
            const int dof = el.GetDof();

            shape.SetSize(dof);
            elvect.SetSize(dof * vdim);
            elvect = 0.0;

            const int int_order = 2 * el.GetOrder() + 2;
            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), int_order);

            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);

               el.CalcPhysShape(T, shape);
               q.Eval(q_vec, T, ip);

               const real_t trans_weight = T.Weight();
               for (int k = 0; k < vdim; k++)
               {
                  const real_t coeff = ip.weight * trans_weight * q_vec(k);
                  for (int s = 0; s < dof; s++)
                  {
                     elvect(dof*k + s) += coeff * shape(s);
                  }
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderVectorDomainLFIntegrator(grad_coef));
      grad_form.Assemble();
   }
};

// =============================================================================
// EXAMPLE: COMPLIANCE OBJECTIVE (for stiffness maximization)
// =============================================================================
// Minimize compliance: J = ∫_0^T ∫_Ω f·u dx dt
// This maximizes structural stiffness under load f.
//
// Usage:
//   VectorCoefficient *load = new MyLoadCoefficient(...);
//   TimeIntegratedObjective *obj = new ComplianceObjective(fes, load, comm);
//
class ComplianceObjective : public TimeIntegratedObjective
{
private:
   VectorCoefficient *applied_load; // non-owning view used in hot paths
   std::unique_ptr<VectorCoefficient> owned_load;

public:
   /// Borrow an externally-owned load coefficient.
   ComplianceObjective(ParFiniteElementSpace *fes,
                       VectorCoefficient &load,
                       MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_), applied_load(&load) {}

   /// Take ownership of a load coefficient.
   ComplianceObjective(ParFiniteElementSpace *fes,
                       std::unique_ptr<VectorCoefficient> load,
                       MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        applied_load(load.get()),
        owned_load(std::move(load)) {}

   /// Backward-compatible constructor for legacy call sites.
   ComplianceObjective(ParFiniteElementSpace *fes,
                       VectorCoefficient *load,
                       MPI_Comm comm_,
                       bool own_load = true)
      : TimeIntegratedObjective(fes, comm_),
        applied_load(load),
        owned_load(own_load ? load : nullptr) {}

   virtual ~ComplianceObjective() = default;

   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                             int timestep, int total_steps) override
   {
      real_t local_work = 0.0;
      Vector u_val, f_val;

      for (int e = 0; e < fespace->GetNE(); e++)
      {
         const FiniteElement *el = fespace->GetFE(e);
         ElementTransformation *T = fespace->GetElementTransformation(e);
         const int int_order = 2 * el->GetOrder() + 2;
         const IntegrationRule &ir = IntRules.Get(el->GetGeomType(), int_order);

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            u.GetVectorValue(*T, ip, u_val);
            applied_load->Eval(f_val, *T, ip);

            local_work += ip.weight * T->Weight() * (f_val * u_val);
         }
      }

      real_t global_work = 0.0;
      MPI_Allreduce(&local_work, &global_work, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      const real_t contribution = TimeWeight(dt, timestep, total_steps) * global_work;
      accumulated_objective += contribution;
      return contribution;
   }

   void ComputeObjectiveGradient(const ParGridFunction &u,
                                 real_t dt, int timestep, int total_steps,
                                 ParLinearForm &grad_form) override
   {
      // ∂J/∂u = f (the applied load)
      const real_t omega = TimeWeight(dt, timestep, total_steps);

      class ScaledLoadCoef : public VectorCoefficient
      {
      private:
         VectorCoefficient *load;
         real_t scale;
      public:
         ScaledLoadCoef(VectorCoefficient *f, real_t s)
            : VectorCoefficient(f->GetVDim()), load(f), scale(s) {}
         void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override
         {
            load->Eval(V, T, ip);
            V *= scale;
         }
      };

      ScaledLoadCoef scaled_load(applied_load, omega);

      class VectorDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape, f_vec;
         VectorCoefficient &f;
      public:
         VectorDomainLFIntegrator(VectorCoefficient &f_) : f(f_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int vdim = f.GetVDim();
            const int dof = el.GetDof();
            shape.SetSize(dof);
            elvect.SetSize(dof * vdim);
            elvect = 0.0;

            const int int_order = 2 * el.GetOrder() + 2;
            const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), int_order);

            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);
               el.CalcPhysShape(T, shape);
               f.Eval(f_vec, T, ip);

               const real_t w = ip.weight * T.Weight();
               for (int k = 0; k < vdim; k++)
               {
                  for (int s = 0; s < dof; s++)
                  {
                     elvect(dof*k + s) += w * f_vec(k) * shape(s);
                  }
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(new VectorDomainLFIntegrator(scaled_load));
      grad_form.Assemble();
   }
};

// =============================================================================
// TODO: Add more objectives here
// =============================================================================
// Examples:
//   - StressL2Objective: minimize ∫∫ |σ(u)|² (stress minimization)
//   - DisplacementTrackingObjective: minimize ∫∫ |u - u_target|²
//   - EnergyObjective: minimize ∫∫ (strain energy or kinetic energy)
//   - TerminalObjective: J = J_T(u(T)) at final time only
//

} // namespace mfem

#endif // OBJECTIVE_FUNCTIONAL_HPP
