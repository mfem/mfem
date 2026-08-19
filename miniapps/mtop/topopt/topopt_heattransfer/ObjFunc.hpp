#ifndef OBJECTIVE_FUNCTIONAL_HPP
#define OBJECTIVE_FUNCTIONAL_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>

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

class RectangularIndicator : public Coefficient
{
private:
   real_t x_min, x_max, y_min, y_max;

public:
   RectangularIndicator(real_t xmin, real_t xmax, real_t ymin, real_t ymax)
      : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(2);
      T.Transform(ip, x);

      return (x(0) >= x_min && x(0) <= x_max &&
              x(1) >= y_min && x(1) <= y_max) ? 1.0 : 0.0;
   }
};

class DoubleRectangularIndicator : public Coefficient
{
private:
   RectangularIndicator first;
   RectangularIndicator second;

public:
   DoubleRectangularIndicator(real_t x1_min, real_t x1_max,
                              real_t y1_min, real_t y1_max,
                              real_t x2_min, real_t x2_max,
                              real_t y2_min, real_t y2_max)
      : first(x1_min, x1_max, y1_min, y1_max),
        second(x2_min, x2_max, y2_min, y2_max) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return std::max(first.Eval(T, ip), second.Eval(T, ip));
   }
};

// Union of three rectangular regions (for passive region specification)
class TripleRectangularIndicator : public Coefficient
{
private:
   std::unique_ptr<Coefficient> first;
   std::unique_ptr<Coefficient> second;
   std::unique_ptr<Coefficient> third;

public:
   TripleRectangularIndicator(std::unique_ptr<Coefficient> first_,
                              std::unique_ptr<Coefficient> second_,
                              std::unique_ptr<Coefficient> third_)
      : first(std::move(first_)),
        second(std::move(second_)),
        third(std::move(third_)) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return std::max({first->Eval(T, ip), second->Eval(T, ip), third->Eval(T, ip)});
   }
};

// =============================================================================
// ABSTRACT BASE CLASS: TerminalObjective
// =============================================================================
// Interface for terminal objective functionals J = ∫_Ω j(u,T) dx
//
// Subclasses must implement:
//   - ComputeObjective: actually computes cost 
//   - ComputeObjectiveGradient: compute ∂J/∂u at one timestep (for adjoint)
//
class HeatTransferObjectiveFunction
{
   protected:
   ParFiniteElementSpace *fespace;
   real_t cost;
   MPI_Comm comm;
   int myid;
   // Trapezoidal rule weights for time integration
   real_t TimeWeight(real_t dt, int timestep, int total_steps) const
   {
      return (timestep == 0 || timestep == total_steps - 1) ? 0.5 * dt : dt;
   }

   public:
   HeatTransferObjectiveFunction(ParFiniteElementSpace *fes, MPI_Comm comm_)
      : fespace(fes), cost(0.0), comm(comm_)
   {
      MPI_Comm_rank(comm, &myid);
   }

   virtual ~HeatTransferObjectiveFunction() = default;

   void Reset() { cost = 0.0; }

   real_t GetObjective() const { return cost; }

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
class TimeIntegratedL2Objective : public HeatTransferObjectiveFunction
{
private:
   Coefficient *subdomain_indicator; // non-owning view used in hot paths
   std::unique_ptr<Coefficient> owned_indicator;

   // One-time setup check: measure of the region the indicator selects. A zero
   // measure means the objective can only ever be zero (e.g. the measurement
   // region is missing from the mesh), which should fail loudly, not silently.
   void CheckIndicatorCoverage()
   {
      real_t local_measure = 0.0;
      for (int e = 0; e < fespace->GetNE(); e++)
      {
         const FiniteElement *el = fespace->GetFE(e);
         ElementTransformation *T = fespace->GetElementTransformation(e);
         const IntegrationRule &ir =
            IntRules.Get(el->GetGeomType(), 2 * el->GetOrder() + 2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            local_measure += ip.weight * T->Weight()
                             * subdomain_indicator->Eval(*T, ip);
         }
      }
      real_t measure = 0.0;
      MPI_Allreduce(&local_measure, &measure, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      if (myid == 0)
      {
         mfem::out << "TimeIntegratedL2Objective: measurement region measure = "
                   << measure << "\n";
         if (measure <= 0.0)
         {
            MFEM_WARNING("TimeIntegratedL2Objective: the indicator selects a "
                         "region of ZERO measure - the objective will be "
                         "identically zero. Check the mesh/indicator.");
         }
      }
   }

public:
   /// Borrow an externally-owned indicator coefficient.
   TimeIntegratedL2Objective(ParFiniteElementSpace *fes,
                           Coefficient &indicator,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(&indicator)
   {
      CheckIndicatorCoverage();
   }

   /// Take ownership of an indicator coefficient.
   TimeIntegratedL2Objective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator.get()),
        owned_indicator(std::move(indicator))
   {
      CheckIndicatorCoverage();
   }

   /// Backward-compatible constructor for legacy call sites.
   TimeIntegratedL2Objective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator),
        owned_indicator(own_indicator ? indicator : nullptr)
   {
      CheckIndicatorCoverage();
   }

   virtual ~TimeIntegratedL2Objective() = default;

   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                             int timestep, int total_steps) override
   {
      real_t local_integral = 0.0;
      real_t u_val;

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
            u_val = u.GetValue(*T, ip);

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
      cost += contribution;

      return contribution;
   }

   void ComputeObjectiveGradient(const ParGridFunction &u,
                                 real_t dt, int timestep, int total_steps,
                                 ParLinearForm &grad_form) override
   {
      const real_t omega = TimeWeight(dt, timestep, total_steps);
      GridFunctionCoefficient u_coef(&u);

      class ObjectiveGradientCoef : public Coefficient
      {
      private:
         GridFunctionCoefficient *u_cf;
         Coefficient *chi;
         real_t weight;

      public:
         ObjectiveGradientCoef(GridFunctionCoefficient *uc,
                               Coefficient *c, real_t w)
            : u_cf(uc), chi(c), weight(w) {}

         real_t Eval(ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            real_t u_val = u_cf->Eval(T, ip);
            const real_t chi_val = chi->Eval(T, ip);
            return 2.0 * weight * chi_val * u_val;
         }
      };

      ObjectiveGradientCoef grad_coef(&u_coef,
                                      subdomain_indicator, omega);

      class HighOrderDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape;
         real_t q_val;
         ObjectiveGradientCoef &q;

      public:
         HighOrderDomainLFIntegrator(ObjectiveGradientCoef &q_)
            : q(q_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int dof = el.GetDof();

            shape.SetSize(dof);
            elvect.SetSize(dof);
            elvect = 0.0;

            const int int_order = 2 * el.GetOrder() + 2;
            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), int_order);

            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);

               el.CalcPhysShape(T, shape);
               q_val = q.Eval(T, ip);

               const real_t trans_weight = T.Weight();
               const real_t coeff = ip.weight * trans_weight * q_val;
               for (int s = 0; s < dof; s++)
               {
                  elvect(s) += coeff * shape(s);
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderDomainLFIntegrator(grad_coef));
      grad_form.Assemble();
   }
};

// =============================================================================
// Target L2 OBJECTIVE: minimize ∫∫ |u(t) - y(t)|² dx dt in subdomain
// =============================================================================
class TimeIntegratedL2TargetObjective : public HeatTransferObjectiveFunction
{
private:
   Coefficient *subdomain_indicator; // non-owning view used in hot paths
   std::unique_ptr<Coefficient> owned_indicator;
   ParGridFunction target;

   // One-time setup check: measure of the region the indicator selects. A zero
   // measure means the objective can only ever be zero (e.g. the measurement
   // region is missing from the mesh), which should fail loudly, not silently.
   void CheckIndicatorCoverage()
   {
      real_t local_measure = 0.0;
      for (int e = 0; e < fespace->GetNE(); e++)
      {
         const FiniteElement *el = fespace->GetFE(e);
         ElementTransformation *T = fespace->GetElementTransformation(e);
         const IntegrationRule &ir =
            IntRules.Get(el->GetGeomType(), 2 * el->GetOrder() + 2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            local_measure += ip.weight * T->Weight()
                             * subdomain_indicator->Eval(*T, ip);
         }
      }
      real_t measure = 0.0;
      MPI_Allreduce(&local_measure, &measure, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      if (myid == 0)
      {
         mfem::out << "TimeIntegratedL2TargetObjective: measurement region measure = "
                   << measure << "\n";
         if (measure <= 0.0)
         {
            MFEM_WARNING("TimeIntegratedL2TargetObjective: the indicator selects a "
                         "region of ZERO measure - the objective will be "
                         "identically zero. Check the mesh/indicator.");
         }
      }
   }

public:
   /// Borrow an externally-owned indicator coefficient.
   TimeIntegratedL2TargetObjective(ParFiniteElementSpace *fes,
                           Coefficient &indicator, ParGridFunction &target_,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(&indicator), target(target_)
   {
      CheckIndicatorCoverage();
   }

   /// Take ownership of an indicator coefficient.
   TimeIntegratedL2TargetObjective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator, ParGridFunction &target_,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator.get()), target(target_),
        owned_indicator(std::move(indicator))
   {
      CheckIndicatorCoverage();
   }

   /// Backward-compatible constructor for legacy call sites.
   TimeIntegratedL2TargetObjective(ParFiniteElementSpace *fes,
                           Coefficient *indicator, ParGridFunction &target_,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator), target(target_),
        owned_indicator(own_indicator ? indicator : nullptr)
   {
      CheckIndicatorCoverage();
   }

   virtual ~TimeIntegratedL2TargetObjective() = default;

   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                             int timestep, int total_steps) override
   {
      real_t local_integral = 0.0;
      real_t u_val;
      real_t target_val;

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
            u_val = u.GetValue(*T, ip);
            target_val = target.GetValue(*T, ip);
            const real_t diff_norm_sq = (u_val - target_val) * (u_val - target_val);
            const real_t chi_val = subdomain_indicator->Eval(*T, ip);
            local_integral += ip.weight * T->Weight() * chi_val * diff_norm_sq;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&local_integral, &global_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      const real_t contribution = TimeWeight(dt, timestep, total_steps)
                                  * global_integral;
      cost += contribution;

      return contribution;
   }

   void ComputeObjectiveGradient(const ParGridFunction &u,
                                 real_t dt, int timestep, int total_steps,
                                 ParLinearForm &grad_form) override
   {
      const real_t omega = TimeWeight(dt, timestep, total_steps);
      GridFunctionCoefficient u_coef(&u);
      GridFunctionCoefficient target_coef(&target);

      class ObjectiveGradientCoef : public Coefficient
      {
      private:
         GridFunctionCoefficient *u_cf;
         Coefficient *chi;
         GridFunctionCoefficient *target_cf;
         real_t weight;

      public:
         ObjectiveGradientCoef(GridFunctionCoefficient *uc, GridFunctionCoefficient *tcf,
                               Coefficient *c, real_t w)
            : u_cf(uc), chi(c), weight(w), target_cf(tcf) {}

         real_t Eval(ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            // u_cf->Eval(T, ip);
            const real_t chi_val = chi->Eval(T, ip);
            real_t diff = u_cf->Eval(T, ip) - target_cf->Eval(T, ip);
            return 2.0 * weight * chi_val * diff;
         }
      };

      
      ObjectiveGradientCoef grad_coef(&u_coef, &target_coef,
                                      subdomain_indicator, omega);

      class HighOrderDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape;
         real_t q_val;
         ObjectiveGradientCoef &q;

      public:
         HighOrderDomainLFIntegrator(ObjectiveGradientCoef &q_)
            : q(q_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int dof = el.GetDof();

            shape.SetSize(dof);
            elvect.SetSize(dof);
            elvect = 0.0;

            const int int_order = 2 * el.GetOrder() + 2;
            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), int_order);

            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);

               el.CalcPhysShape(T, shape);
               q_val = q.Eval(T, ip);

               const real_t trans_weight = T.Weight();
               const real_t coeff = ip.weight * trans_weight * q_val;
               for (int s = 0; s < dof; s++)
               {
                  elvect(s) += coeff * shape(s);
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderDomainLFIntegrator(grad_coef));
      grad_form.Assemble();
   }
};

// =============================================================================
// Terminal L2 OBJECTIVE: minimize ∫ |u(x)|² dx in subdomain
// =============================================================================
class TerminalL2Objective : public HeatTransferObjectiveFunction
{
   protected:
   Coefficient *subdomain_indicator; // non-owning view used in hot paths
   std::unique_ptr<Coefficient> owned_indicator;

   // One-time setup check: measure of the region the indicator selects. A zero
   // measure means the objective can only ever be zero (e.g. the measurement
   // region is missing from the mesh), which should fail loudly, not silently.
   void CheckIndicatorCoverage()
   {
      real_t local_measure = 0.0;
      for (int e = 0; e < fespace->GetNE(); e++)
      {
         const FiniteElement *el = fespace->GetFE(e);
         ElementTransformation *T = fespace->GetElementTransformation(e);
         const IntegrationRule &ir =
            IntRules.Get(el->GetGeomType(), 2 * el->GetOrder() + 2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            local_measure += ip.weight * T->Weight()
                             * subdomain_indicator->Eval(*T, ip);
         }
      }
      real_t measure = 0.0;
      MPI_Allreduce(&local_measure, &measure, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      if (myid == 0)
      {
         mfem::out << "TerminalL2Objective: measurement region measure = "
                   << measure << "\n";
         if (measure <= 0.0)
         {
            MFEM_WARNING("TerminalL2Objective: the indicator selects a "
                         "region of ZERO measure - the objective will be "
                         "identically zero. Check the mesh/indicator.");
         }
      }
   }

   public:
   /// Borrow an externally-owned indicator coefficient.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           Coefficient &indicator,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(&indicator) 
        {
            CheckIndicatorCoverage();
        }

   /// Take ownership of an indicator coefficient.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator.get()),
        owned_indicator(std::move(indicator)) 
        {
            CheckIndicatorCoverage();
        }

   /// Backward-compatible constructor for legacy call sites.
   TerminalL2Objective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator),
        owned_indicator(own_indicator ? indicator : nullptr) 
        {
            CheckIndicatorCoverage();
        }

   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                             int timestep, int total_steps) override
   {
      // ConstantCoefficient zero(0.0);
      // cost = u.ComputeL2Error(zero)*u.ComputeL2Error(zero);
      if (timestep != total_steps - 1)
      {
         cost = 0.0;
         return 0.0;
      }
      else
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

         cost = global_integral;
         return global_integral;
      }
   }

   virtual ~TerminalL2Objective() = default;

   void ComputeObjectiveGradient(const ParGridFunction &u, real_t dt, 
      int timestep, int total_steps, ParLinearForm &grad_form) override
   {
      if (timestep != total_steps - 1)
      {
         ConstantCoefficient zero(0.0);
         grad_form.AddDomainIntegrator(new DomainLFIntegrator(zero));
         grad_form.Assemble();
      }
      else
      {
         GridFunctionCoefficient u_coef(&u);
         class ObjectiveGradientCoef : public Coefficient
         {
         private:
            GridFunctionCoefficient *u_cf;
            Coefficient *chi;

         public:
            ObjectiveGradientCoef(GridFunctionCoefficient *uc, Coefficient *c)
               : u_cf(uc), chi(c) {}

            real_t Eval(ElementTransformation &T,
                     const IntegrationPoint &ip) override
            {
               const real_t chi_val = chi->Eval(T, ip);
               return 2.0 * chi_val * (u_cf->Eval(T, ip));
            }
         };

         ObjectiveGradientCoef grad_coef(&u_coef, subdomain_indicator);

         class HighOrderDomainLFIntegrator : public LinearFormIntegrator
         {
         private:
            Vector shape; 
            real_t q_val; 
            ObjectiveGradientCoef &q;

         public:
            HighOrderDomainLFIntegrator(ObjectiveGradientCoef &q_)
               : q(q_) {}

            void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &T,
                                       Vector &elvect) override
            {
               const int dof = el.GetDof();

               shape.SetSize(dof);
               elvect.SetSize(dof);
               elvect = 0.0;

               const int int_order = 2 * el.GetOrder() + 2;
               const IntegrationRule &ir =
                  IntRules.Get(el.GetGeomType(), int_order);

               for (int i = 0; i < ir.GetNPoints(); i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  T.SetIntPoint(&ip);

                  el.CalcPhysShape(T, shape);
                  q_val = q.Eval(T, ip);
                  const real_t trans_weight = T.Weight();
                  const real_t coeff = ip.weight * trans_weight * q_val;
                  for (int s = 0; s < dof; s++)
                  {
                     elvect(s) += coeff * shape(s);
                  }
               }
            }

            using LinearFormIntegrator::AssembleRHSElementVect;
         };
         
         grad_form.AddDomainIntegrator(
            new HighOrderDomainLFIntegrator(grad_coef));
         grad_form.Assemble();
         // if (timestep != total_steps - 1)
         // {
         //    MFEM_WARNING("Computing Gradient for Terminal Objective at Intermediate Time!");
         // }
      }
   }
};

// =============================================================================
// Terminal Target OBJECTIVE: minimize ∫ |u(x) - y|² dx in subdomain
// =============================================================================
class TerminalTargetObjective : public HeatTransferObjectiveFunction
{
   protected:
   Coefficient *subdomain_indicator; // non-owning view used in hot paths
   std::unique_ptr<Coefficient> owned_indicator;
   ParGridFunction target;
   
   // One-time setup check: measure of the region the indicator selects. A zero
   // measure means the objective can only ever be zero (e.g. the measurement
   // region is missing from the mesh), which should fail loudly, not silently.
   void CheckIndicatorCoverage()
   {
      real_t local_measure = 0.0;
      for (int e = 0; e < fespace->GetNE(); e++)
      {
         const FiniteElement *el = fespace->GetFE(e);
         ElementTransformation *T = fespace->GetElementTransformation(e);
         const IntegrationRule &ir =
            IntRules.Get(el->GetGeomType(), 2 * el->GetOrder() + 2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            local_measure += ip.weight * T->Weight()
                             * subdomain_indicator->Eval(*T, ip);
         }
      }
      real_t measure = 0.0;
      MPI_Allreduce(&local_measure, &measure, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      if (myid == 0)
      {
         mfem::out << "TerminaTargetL2Objective: measurement region measure = "
                   << measure << "\n";
         if (measure <= 0.0)
         {
            MFEM_WARNING("TerminalTargetL2Objective: the indicator selects a "
                         "region of ZERO measure - the objective will be "
                         "identically zero. Check the mesh/indicator.");
         }
      }
   }

   public:
   /// Borrow an externally-owned indicator coefficient.
   TerminalTargetObjective(ParFiniteElementSpace *fes,
                           Coefficient &indicator,  ParGridFunction &target_,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_), target(target_),
        subdomain_indicator(&indicator) 
        {
            CheckIndicatorCoverage();
        }

   /// Take ownership of an indicator coefficient.
   TerminalTargetObjective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator, ParGridFunction &target_,
                           MPI_Comm comm_)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator.get()), target(target_),
        owned_indicator(std::move(indicator)) 
        {
            CheckIndicatorCoverage();
        }

   /// Backward-compatible constructor for legacy call sites.
   TerminalTargetObjective(ParFiniteElementSpace *fes,
                           Coefficient *indicator, ParGridFunction &target_,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : HeatTransferObjectiveFunction(fes, comm_),
        subdomain_indicator(indicator), target(target_),
        owned_indicator(own_indicator ? indicator : nullptr) 
        {
            CheckIndicatorCoverage();
        }

   real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                             int timestep, int total_steps) override
   {
      if (timestep != total_steps - 1)
      {
         cost = 0.0;
         return 0.0;
      }
      else
      {
         real_t local_integral = 0.0;
         real_t target_val;
         real_t u_val;

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
               u_val = u.GetValue(*T, ip);
               target_val = target.GetValue(*T, ip);
               const real_t diff_norm_sq = (u_val - target_val) * (u_val - target_val);
               const real_t chi_val = subdomain_indicator->Eval(*T, ip);
               local_integral += ip.weight * T->Weight() * chi_val * diff_norm_sq;
            }
         }

         real_t global_integral = 0.0;
         MPI_Allreduce(&local_integral, &global_integral, 1,
                     MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

         cost = global_integral;
         return global_integral;
      }
   }

   virtual ~TerminalTargetObjective() = default;

   void ComputeObjectiveGradient(const ParGridFunction &u, real_t dt, int timestep, int total_steps, 
      ParLinearForm &grad_form) override
   {
      if (timestep != total_steps - 1)
      {
         ConstantCoefficient zero(0.0);
         grad_form.AddDomainIntegrator(new DomainLFIntegrator(zero));
         grad_form.Assemble();
      }
      else
      {
         GridFunctionCoefficient u_coef(&u);
         GridFunctionCoefficient target_coef(&target);
         class ObjectiveGradientCoef : public Coefficient
         {
         private:
            GridFunctionCoefficient *u_cf;
            GridFunctionCoefficient *target_cf;
            Coefficient *chi;

         public:
            ObjectiveGradientCoef(GridFunctionCoefficient *uc, GridFunctionCoefficient *tc, Coefficient *c)
               : u_cf(uc), target_cf(tc), chi(c) {}

            real_t Eval(ElementTransformation &T,
                     const IntegrationPoint &ip) override
            {
               const real_t chi_val = chi->Eval(T, ip);
               real_t diff = u_cf->Eval(T, ip) - target_cf->Eval(T, ip);
               return 2.0 * chi_val * diff;
            }
         };

         ObjectiveGradientCoef grad_coef(&u_coef, &target_coef, subdomain_indicator);

         class HighOrderDomainLFIntegrator : public LinearFormIntegrator
         {
         private:
            Vector shape; 
            real_t q_val; 
            ObjectiveGradientCoef &q;

         public:
            HighOrderDomainLFIntegrator(ObjectiveGradientCoef &q_)
               : q(q_) {}

            void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &T,
                                       Vector &elvect) override
            {
               const int dof = el.GetDof();

               shape.SetSize(dof);
               elvect.SetSize(dof);
               elvect = 0.0;

               const int int_order = 2 * el.GetOrder() + 2;
               const IntegrationRule &ir =
                  IntRules.Get(el.GetGeomType(), int_order);

               for (int i = 0; i < ir.GetNPoints(); i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  T.SetIntPoint(&ip);

                  el.CalcPhysShape(T, shape);
                  q_val = q.Eval(T, ip);
                  const real_t trans_weight = T.Weight();
                  const real_t coeff = ip.weight * trans_weight * q_val;
                  for (int s = 0; s < dof; s++)
                  {
                     elvect(s) += coeff * shape(s);
                  }
               }
            }

            using LinearFormIntegrator::AssembleRHSElementVect;
         };

         grad_form.AddDomainIntegrator(
            new HighOrderDomainLFIntegrator(grad_coef));
         grad_form.Assemble();
         // if (timestep != total_steps - 1)
         // {
         //    MFEM_WARNING("Computing Gradient for Terminal Objective at Intermediate Time!");
         // }
      }
   }
};

}
#endif 