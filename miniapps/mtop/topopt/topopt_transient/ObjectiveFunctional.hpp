// =============================================================================
// Objective Functional for Transient Topology Optimization
// =============================================================================
//
// Minimize displacement in a protected subdomain:
//   J = integral_0^T integral_{Omega_hat} |u(t)|^2 dx dt
//
// =============================================================================

#ifndef OBJECTIVE_FUNCTIONAL_HPP
#define OBJECTIVE_FUNCTIONAL_HPP

#include "mfem.hpp"
#include <cmath>

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

class TimeIntegratedObjective
{
private:
   ParFiniteElementSpace *fespace;
   Coefficient *subdomain_indicator;

   real_t accumulated_objective;

   MPI_Comm comm;
   int myid;

   real_t TimeWeight(real_t dt, int timestep, int total_steps) const
   {
      return (timestep == 0 || timestep == total_steps - 1) ? 0.5 * dt : dt;
   }

public:
   TimeIntegratedObjective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_)
      : fespace(fes), subdomain_indicator(indicator),
        accumulated_objective(0.0), comm(comm_)
   {
      MPI_Comm_rank(comm, &myid);
   }

   void Reset() { accumulated_objective = 0.0; }

   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                             int timestep, int total_steps)
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

   real_t GetObjective() const { return accumulated_objective; }

   void ComputeObjectiveGradient(const ParGridFunction &u,
                                 real_t dt, int timestep, int total_steps,
                                 ParLinearForm &grad_form)
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

} // namespace mfem

#endif // OBJECTIVE_FUNCTIONAL_HPP
