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
#include "BoundaryTraceHistory.hpp"
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

// Indicator for an axis-aligned box in 3D.
class BoxIndicator3D : public Coefficient
{
private:
   real_t x_min, x_max, y_min, y_max, z_min, z_max;

public:
   BoxIndicator3D(real_t xmin, real_t xmax,
                  real_t ymin, real_t ymax,
                  real_t zmin, real_t zmax)
      : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax),
        z_min(zmin), z_max(zmax) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);

      return (x(0) >= x_min && x(0) <= x_max &&
              x(1) >= y_min && x(1) <= y_max &&
              x(2) >= z_min && x(2) <= z_max) ? 1.0 : 0.0;
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
// 3D SPHERICAL INDICATOR COEFFICIENTS
// =============================================================================

// Indicator for a solid sphere (r < radius)
class SphericalIndicator : public Coefficient
{
private:
   real_t radius;
   Vector center;

public:
   SphericalIndicator(real_t r, const Vector *ctr = nullptr)
      : radius(r)
   {
      center.SetSize(3);
      if (ctr && ctr->Size() == 3) { center = *ctr; }
      else { center = 0.0; }
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);
      x -= center;
      real_t r = x.Norml2();
      return (r < radius) ? 1.0 : 0.0;
   }
};

// Indicator for a spherical shell (r_inner < r < r_outer)
class SphericalShellIndicator : public Coefficient
{
private:
   real_t r_inner, r_outer;
   Vector center;

public:
   SphericalShellIndicator(real_t r_in, real_t r_out, const Vector *ctr = nullptr)
      : r_inner(r_in), r_outer(r_out)
   {
      center.SetSize(3);
      if (ctr && ctr->Size() == 3) { center = *ctr; }
      else { center = 0.0; }
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);
      x -= center;
      real_t r = x.Norml2();
      return (r > r_inner && r < r_outer) ? 1.0 : 0.0;
   }
};

// Indicator for union of multiple spherical shells
class MultiSphericalShellIndicator : public Coefficient
{
private:
   std::vector<std::pair<real_t, real_t>> shell_ranges;
   Vector center;

public:
   MultiSphericalShellIndicator(const Vector *ctr = nullptr)
   {
      center.SetSize(3);
      if (ctr && ctr->Size() == 3) { center = *ctr; }
      else { center = 0.0; }
   }

   void AddShell(real_t r_in, real_t r_out)
   {
      shell_ranges.push_back({r_in, r_out});
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(3);
      T.Transform(ip, x);
      x -= center;
      real_t r = x.Norml2();

      for (const auto &range : shell_ranges)
      {
         if (r > range.first && r < range.second) return 1.0;
      }
      return 0.0;
   }
};

// =============================================================================
// ABSTRACT BASE CLASS: TimeIntegratedObjective
// =============================================================================
// Interface for time-integrated objective functionals J = ∫_0^T ∫_Ω j(u,t) dx dt
//
// Subclasses provide instantaneous spatial values/gradients.  The legacy
// discrete wrappers retain trapezoidal time weights for the exact discrete
// RK4-adjoint path; continuous time integrators call the instantaneous API.
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

   /// Shared spatial assembly used by both instantaneous and discrete paths.
   /// The discrete adapter passes its trapezoidal scale here so multiplication
   /// stays inside the element quadrature, preserving the historical arithmetic.
   virtual void AssembleStateGradientScaled(
      const ParGridFunction &u, real_t time, real_t scale,
      ParLinearForm &grad_form) = 0;

public:
   TimeIntegratedObjective(ParFiniteElementSpace *fes, MPI_Comm comm_)
      : fespace(fes), accumulated_objective(0.0), comm(comm_)
   {
      MPI_Comm_rank(comm, &myid);
   }

   virtual ~TimeIntegratedObjective() = default;

   void Reset() { accumulated_objective = 0.0; }
   real_t GetObjective() const { return accumulated_objective; }

   /// Instantaneous spatial integrand ell(u,t), with no time-quadrature weight.
   virtual real_t EvaluateInstantaneous(const ParGridFunction &u,
                                        real_t time) = 0;

   /// Assemble d ell/du at physical time t, with no time-quadrature weight.
   void AssembleInstantaneousStateGradient(
      const ParGridFunction &u, real_t time,
      ParLinearForm &grad_form)
   {
      AssembleStateGradientScaled(u, time, 1.0, grad_form);
   }

   /// Accumulate objective contribution at current timestep
   /// @return contribution added (for monitoring)
   real_t AccumulateTimestepAtTime(const ParGridFunction &u,
                                   real_t physical_time,
                                   real_t dt,
                                   int timestep,
                                   int total_steps)
   {
      const real_t contribution = TimeWeight(dt, timestep, total_steps)
                                  * EvaluateInstantaneous(u, physical_time);
      accumulated_objective += contribution;
      return contribution;
   }

   /// Legacy zero-start adapter.
   virtual real_t AccumulateTimestep(const ParGridFunction &u, real_t dt,
                                      int timestep, int total_steps)
   {
      return AccumulateTimestepAtTime(
         u, timestep * dt, dt, timestep, total_steps);
   }

   /// Compute objective gradient ∂J/∂u at current timestep (for adjoint)
   void ComputeObjectiveGradientAtTime(
      const ParGridFunction &u, real_t physical_time,
      real_t dt, int timestep, int total_steps,
      ParLinearForm &grad_form)
   {
      AssembleStateGradientScaled(
         u, physical_time, TimeWeight(dt, timestep, total_steps), grad_form);
   }

   /// Legacy zero-start adapter.
   virtual void ComputeObjectiveGradient(const ParGridFunction &u,
                                         real_t dt, int timestep, int total_steps,
                                         ParLinearForm &grad_form)
   {
      ComputeObjectiveGradientAtTime(
         u, timestep * dt, dt, timestep, total_steps, grad_form);
   }
};

// =============================================================================
// BOUNDARY DISPLACEMENT TRACKING
// =============================================================================
//
//   ell(u,t) = 1/2 int_Gamma_obs |u-u_dagger(t)|^2 ds.
//
// Reference traces are sampled on the reconstruction state space at every
// coarse half step.  Consequently both RK4 midpoint stages request the same
// physical observation, and reverse-time adjoint access follows exactly the
// same integer-indexed lookup path as the forward objective.
class BoundaryDisplacementTrackingObjective : public TimeIntegratedObjective
{
private:
   std::shared_ptr<const BoundaryTraceHistory> trace_history_;
   Array<int> observation_marker_;
   real_t observed_boundary_measure_;

   void CheckBoundaryCoverage()
   {
      ParMesh *pmesh = fespace->GetParMesh();
      real_t local_measure = 0.0;
      for (int be = 0; be < pmesh->GetNBE(); be++)
      {
         const int attribute = pmesh->GetBdrAttribute(be);
         if (observation_marker_[attribute - 1] == 0) { continue; }

         const FiniteElement *el = fespace->GetBE(be);
         ElementTransformation *T =
            fespace->GetBdrElementTransformation(be);
         const IntegrationRule &ir =
            IntRules.Get(el->GetGeomType(), 2 * el->GetOrder() + 2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            local_measure += ip.weight * T->Weight();
         }
      }

      MPI_Allreduce(&local_measure, &observed_boundary_measure_, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      MFEM_VERIFY(std::isfinite(observed_boundary_measure_) &&
                  observed_boundary_measure_ > 0.0,
                  "BoundaryDisplacementTrackingObjective marker selects an "
                  "empty boundary.");
      if (myid == 0)
      {
         mfem::out << "BoundaryDisplacementTrackingObjective: observed "
                   << "boundary measure = " << observed_boundary_measure_
                   << "\n";
      }
   }

protected:
   void AssembleStateGradientScaled(
      const ParGridFunction &u, real_t time, real_t scale,
      ParLinearForm &grad_form) override
   {
      const ParGridFunction &target = trace_history_->GetSampleAtTime(time);
      VectorGridFunctionCoefficient u_coefficient(&u);
      VectorGridFunctionCoefficient target_coefficient(&target);

      class TrackingDifferenceCoefficient : public VectorCoefficient
      {
      private:
         VectorGridFunctionCoefficient &u_;
         VectorGridFunctionCoefficient &target_;
         real_t scale_;
         Vector target_value_;

      public:
         TrackingDifferenceCoefficient(
            int vdim, VectorGridFunctionCoefficient &u,
            VectorGridFunctionCoefficient &target, real_t scale)
            : VectorCoefficient(vdim), u_(u), target_(target), scale_(scale) {}

         void Eval(Vector &value, ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            u_.Eval(value, T, ip);
            target_.Eval(target_value_, T, ip);
            value -= target_value_;
            value *= scale_;
         }
      };

      // MFEM's stock VectorBoundaryLFIntegrator defaults to order 2*p.  The
      // inverse objective uses 2*p+2 for both value and derivative, including
      // on variable-order spaces, so select the rule from each boundary FE.
      class HighOrderVectorBoundaryLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape_, value_;
         VectorCoefficient &coefficient_;

      public:
         explicit HighOrderVectorBoundaryLFIntegrator(
            VectorCoefficient &coefficient)
            : coefficient_(coefficient) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int vdim = coefficient_.GetVDim();
            const int dof = el.GetDof();
            shape_.SetSize(dof);
            elvect.SetSize(dof * vdim);
            elvect = 0.0;

            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 2);
            for (int q = 0; q < ir.GetNPoints(); q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               T.SetIntPoint(&ip);
               el.CalcShape(ip, shape_);
               coefficient_.Eval(value_, T, ip);
               const real_t weight = ip.weight * T.Weight();
               for (int component = 0; component < vdim; component++)
               {
                  const real_t scaled_value =
                     weight * value_[component];
                  for (int basis = 0; basis < dof; basis++)
                  {
                     elvect[dof * component + basis] +=
                        scaled_value * shape_[basis];
                  }
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      TrackingDifferenceCoefficient difference(
         u.VectorDim(), u_coefficient, target_coefficient, scale);
      grad_form.AddBoundaryIntegrator(
         new HighOrderVectorBoundaryLFIntegrator(difference),
         observation_marker_);
      grad_form.Assemble();
   }

public:
   BoundaryDisplacementTrackingObjective(
      ParFiniteElementSpace *fes,
      std::shared_ptr<const BoundaryTraceHistory> trace_history,
      MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        trace_history_(std::move(trace_history)),
        observed_boundary_measure_(0.0)
   {
      MFEM_VERIFY(trace_history_,
                  "BoundaryDisplacementTrackingObjective requires reference "
                  "trace data.");
      MFEM_VERIFY(trace_history_->FESpace() == fespace,
                  "BoundaryDisplacementTrackingObjective trace data belongs "
                  "to a different finite element space.");
      trace_history_->ValidateComplete();
      observation_marker_ = trace_history_->ObservationMarker();
      CheckBoundaryCoverage();
   }

   real_t ObservedBoundaryMeasure() const
   {
      return observed_boundary_measure_;
   }

   const BoundaryTraceHistory &TraceHistory() const
   {
      return *trace_history_;
   }

   real_t EvaluateInstantaneous(const ParGridFunction &u,
                                real_t time) override
   {
      MFEM_VERIFY(u.ParFESpace() == fespace,
                  "BoundaryDisplacementTrackingObjective received a state "
                  "from a different finite element space.");
      const ParGridFunction &target = trace_history_->GetSampleAtTime(time);
      ParMesh *pmesh = fespace->GetParMesh();
      real_t local_integral = 0.0;
      Vector u_value, target_value;

      for (int be = 0; be < pmesh->GetNBE(); be++)
      {
         const int attribute = pmesh->GetBdrAttribute(be);
         if (observation_marker_[attribute - 1] == 0) { continue; }

         const FiniteElement *el = fespace->GetBE(be);
         ElementTransformation *T =
            fespace->GetBdrElementTransformation(be);
         const IntegrationRule &ir =
            IntRules.Get(el->GetGeomType(), 2 * el->GetOrder() + 2);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            u.GetVectorValue(*T, ip, u_value);
            target.GetVectorValue(*T, ip, target_value);
            u_value -= target_value;
            local_integral +=
               0.5 * ip.weight * T->Weight() * (u_value * u_value);
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&local_integral, &global_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
      MFEM_VERIFY(std::isfinite(global_integral),
                  "Boundary displacement tracking objective is non-finite.");
      return global_integral;
   }
};

// Evaluate the instantaneous spatial inner product <mode, u>.  This is a
// diagnostic observable, not a time-integrated objective: it intentionally
// carries no timestep-dependent quadrature weight and has no adjoint effect.
// A supplied mode can therefore monitor forward temporal convergence without
// changing the objective used to define the adjoint RHS.
inline real_t EvaluateDisplacementModalProjection(
   ParFiniteElementSpace &fespace, const ParGridFunction &u,
   VectorCoefficient &mode, MPI_Comm comm)
{
   real_t local_projection = 0.0;
   Vector u_value, mode_value;

   for (int e = 0; e < fespace.GetNE(); e++)
   {
      const FiniteElement *el = fespace.GetFE(e);
      ElementTransformation *T = fespace.GetElementTransformation(e);
      const IntegrationRule &ir =
         IntRules.Get(el->GetGeomType(), 2 * el->GetOrder() + 2);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         u.GetVectorValue(*T, ip, u_value);
         mode.Eval(mode_value, *T, ip);
         local_projection += ip.weight * T->Weight() * (mode_value * u_value);
      }
   }

   real_t global_projection = 0.0;
   MPI_Allreduce(&local_projection, &global_projection, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return global_projection;
}

// =============================================================================
// DISPLACEMENT L2 OBJECTIVE: minimize ∫∫ |u(t)|² dx dt in subdomain
// =============================================================================
class DisplacementL2Objective : public TimeIntegratedObjective
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
         mfem::out << "DisplacementL2Objective: measurement region measure = "
                   << measure << "\n";
         if (measure <= 0.0)
         {
            MFEM_WARNING("DisplacementL2Objective: the indicator selects a "
                         "region of ZERO measure - the objective will be "
                         "identically zero. Check the mesh/indicator.");
         }
      }
   }

public:
   /// Borrow an externally-owned indicator coefficient.
   DisplacementL2Objective(ParFiniteElementSpace *fes,
                           Coefficient &indicator,
                           MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        subdomain_indicator(&indicator)
   {
      CheckIndicatorCoverage();
   }

   /// Take ownership of an indicator coefficient.
   DisplacementL2Objective(ParFiniteElementSpace *fes,
                           std::unique_ptr<Coefficient> indicator,
                           MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        subdomain_indicator(indicator.get()),
        owned_indicator(std::move(indicator))
   {
      CheckIndicatorCoverage();
   }

   /// Backward-compatible constructor for legacy call sites.
   DisplacementL2Objective(ParFiniteElementSpace *fes,
                           Coefficient *indicator,
                           MPI_Comm comm_,
                           bool own_indicator = true)
      : TimeIntegratedObjective(fes, comm_),
        subdomain_indicator(indicator),
        owned_indicator(own_indicator ? indicator : nullptr)
   {
      CheckIndicatorCoverage();
   }

   virtual ~DisplacementL2Objective() = default;

   real_t EvaluateInstantaneous(const ParGridFunction &u,
                                real_t time) override
   {
      (void)time;
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

      return global_integral;
   }

   void AssembleStateGradientScaled(
      const ParGridFunction &u, real_t time, real_t scale,
      ParLinearForm &grad_form) override
   {
      (void)time;
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
                                      subdomain_indicator, scale);

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
// HARMONIC DISPLACEMENT TRACKING
// =============================================================================
// J = int_0^T int_R |u(x,t) - A cos(2*pi*f*t+phase) psi(x)|^2 dx dt.
//
// At a zero state, dJ/du is a scaled copy of psi.  This makes the objective a
// useful, physically interpretable way to prescribe the spatial spectrum of
// the adjoint source while leaving the forward load independently selectable.
class HarmonicDisplacementTrackingObjective : public TimeIntegratedObjective
{
private:
   Coefficient *region;
   VectorCoefficient *target_mode;
   std::unique_ptr<Coefficient> owned_region;
   std::unique_ptr<VectorCoefficient> owned_target_mode;
   real_t target_amplitude;
   real_t target_frequency;
   real_t target_phase;

   real_t TargetFactor(real_t time) const
   {
      constexpr real_t two_pi =
         2.0 * 3.1415926535897932384626433832795;
      return target_amplitude *
             std::cos(two_pi * target_frequency * time + target_phase);
   }

public:
   HarmonicDisplacementTrackingObjective(
      ParFiniteElementSpace *fes,
      std::unique_ptr<Coefficient> region_,
      std::unique_ptr<VectorCoefficient> target_mode_,
      real_t amplitude, real_t frequency, real_t phase,
      MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        region(region_.get()), target_mode(target_mode_.get()),
        owned_region(std::move(region_)),
        owned_target_mode(std::move(target_mode_)),
        target_amplitude(amplitude),
        target_frequency(frequency),
        target_phase(phase) {}

   real_t EvaluateInstantaneous(const ParGridFunction &u,
                                real_t time) override
   {
      const real_t target_factor = TargetFactor(time);
      real_t local_integral = 0.0;
      Vector u_value, target_value;

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
            const real_t chi = region->Eval(*T, ip);
            if (chi == 0.0) { continue; }

            u.GetVectorValue(*T, ip, u_value);
            target_mode->Eval(target_value, *T, ip);
            target_value *= target_factor;
            u_value -= target_value;
            local_integral += ip.weight * T->Weight() * chi *
                              (u_value * u_value);
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&local_integral, &global_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);

      return global_integral;
   }

   void AssembleStateGradientScaled(
      const ParGridFunction &u, real_t time, real_t scale,
      ParLinearForm &grad_form) override
   {
      const real_t target_factor = TargetFactor(time);
      VectorGridFunctionCoefficient u_coef(&u);

      class TrackingGradientCoefficient : public VectorCoefficient
      {
      private:
         VectorGridFunctionCoefficient &u;
         VectorCoefficient &target;
         Coefficient &region;
         real_t target_factor;
         real_t scale;
         Vector target_value;

      public:
         TrackingGradientCoefficient(int vdim,
                                     VectorGridFunctionCoefficient &u_,
                                     VectorCoefficient &target_,
                                     Coefficient &region_,
                                     real_t target_factor_,
                                     real_t scale_)
            : VectorCoefficient(vdim), u(u_), target(target_),
              region(region_), target_factor(target_factor_), scale(scale_) {}

         void Eval(Vector &value, ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            u.Eval(value, T, ip);
            target.Eval(target_value, T, ip);
            value.Add(-target_factor, target_value);
            value *= 2.0 * scale * region.Eval(T, ip);
         }
      };

      TrackingGradientCoefficient gradient(
         u.VectorDim(), u_coef, *target_mode, *region,
         target_factor, scale);

      // Match EvaluateInstantaneous's explicit 2*p+2 spatial quadrature.
      // MFEM's stock VectorDomainLFIntegrator defaults to a lower rule, which
      // is not generally the exact derivative when the region/mode varies.
      class HighOrderVectorDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape, value;
         VectorCoefficient &coefficient;

      public:
         HighOrderVectorDomainLFIntegrator(VectorCoefficient &coefficient_)
            : coefficient(coefficient_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int vdim = coefficient.GetVDim();
            const int dof = el.GetDof();
            shape.SetSize(dof);
            elvect.SetSize(dof * vdim);
            elvect = 0.0;

            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 2);
            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);
               el.CalcPhysShape(T, shape);
               coefficient.Eval(value, T, ip);

               const real_t weight = ip.weight * T.Weight();
               for (int component = 0; component < vdim; component++)
               {
                  const real_t scaled_value =
                     weight * value(component);
                  for (int basis = 0; basis < dof; basis++)
                  {
                     elvect(dof * component + basis) +=
                        scaled_value * shape(basis);
                  }
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderVectorDomainLFIntegrator(gradient));
      grad_form.Assemble();
   }
};

// =============================================================================
// HARMONIC MODAL CORRELATION
// =============================================================================
// J = -int_0^T A cos(2*pi*f*t+phase) <psi,u> dt.
//
// Minimizing J maximizes the in-phase displacement in psi.  Unlike pointwise
// tracking, dJ/du is independent of the current state, so a low-mode psi stays
// a genuinely low-spectrum adjoint source even when the forward state contains
// high modes.  This is useful for the reversed fine-forward/coarse-adjoint
// allocation experiment.
class HarmonicModalCorrelationObjective : public TimeIntegratedObjective
{
private:
   VectorCoefficient *target_mode;
   std::unique_ptr<VectorCoefficient> owned_target_mode;
   real_t target_amplitude;
   real_t target_frequency;
   real_t target_phase;

   real_t TargetFactor(real_t time) const
   {
      constexpr real_t two_pi =
         2.0 * 3.1415926535897932384626433832795;
      return target_amplitude *
             std::cos(two_pi * target_frequency * time + target_phase);
   }

public:
   HarmonicModalCorrelationObjective(
      ParFiniteElementSpace *fes,
      std::unique_ptr<VectorCoefficient> target_mode_,
      real_t amplitude, real_t frequency, real_t phase,
      MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        target_mode(target_mode_.get()),
        owned_target_mode(std::move(target_mode_)),
        target_amplitude(amplitude), target_frequency(frequency),
        target_phase(phase) {}

   real_t EvaluateInstantaneous(const ParGridFunction &u,
                                real_t time) override
   {
      return -TargetFactor(time) *
             EvaluateDisplacementModalProjection(
                *fespace, u, *target_mode, comm);
   }

   void AssembleStateGradientScaled(
      const ParGridFunction &u, real_t time, real_t scale,
      ParLinearForm &grad_form) override
   {
      (void)u;

      class ScaledModeCoefficient : public VectorCoefficient
      {
      private:
         VectorCoefficient &mode;
         real_t factor;

      public:
         ScaledModeCoefficient(VectorCoefficient &mode_, real_t factor_)
            : VectorCoefficient(mode_.GetVDim()), mode(mode_), factor(factor_) {}

         void Eval(Vector &value, ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            mode.Eval(value, T, ip);
            value *= factor;
         }
      };

      ScaledModeCoefficient gradient(
         *target_mode, -scale * TargetFactor(time));

      class HighOrderVectorDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape, mode_value;
         VectorCoefficient &mode;

      public:
         explicit HighOrderVectorDomainLFIntegrator(VectorCoefficient &mode_)
            : mode(mode_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int vdim = mode.GetVDim();
            const int dof = el.GetDof();
            shape.SetSize(dof);
            elvect.SetSize(dof * vdim);
            elvect = 0.0;

            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 2);
            for (int q = 0; q < ir.GetNPoints(); q++)
            {
               const IntegrationPoint &ip = ir.IntPoint(q);
               T.SetIntPoint(&ip);
               el.CalcPhysShape(T, shape);
               mode.Eval(mode_value, T, ip);
               const real_t weight = ip.weight * T.Weight();
               for (int component = 0; component < vdim; component++)
               {
                  for (int basis = 0; basis < dof; basis++)
                  {
                     elvect(dof * component + basis) +=
                        weight * mode_value(component) * shape(basis);
                  }
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderVectorDomainLFIntegrator(gradient));
      grad_form.Assemble();
   }
};

// =============================================================================
// WINDOWED MODAL-ENERGY OBJECTIVE
// =============================================================================
//
// Let m_j(t)=<psi_j,u(t)> be output-modal amplitudes. This functional
//
//   J = 1/2 int w(t) [ beta_0 m_0(t)^2 - beta_T m_T(t)^2 ] dt
//
// penalizes residual source-mode energy and rewards converted-mode energy.
// Unlike harmonic correlation, its state gradient depends on the instantaneous
// forward state. It is therefore a deliberately stage-sensitive same-grid RK4
// experiment for the DO/OD comparison.
class WindowedModalEnergyObjective : public TimeIntegratedObjective
{
private:
   VectorCoefficient *converted_mode;
   VectorCoefficient *residual_mode;
   std::unique_ptr<VectorCoefficient> owned_converted_mode;
   std::unique_ptr<VectorCoefficient> owned_residual_mode;
   real_t converted_weight;
   real_t residual_weight;
   real_t window_start;
   real_t window_ramp;

   real_t Window(real_t time) const
   {
      if (time <= window_start) { return 0.0; }
      if (window_ramp <= 0.0) { return 1.0; }
      const real_t xi = (time - window_start) / window_ramp;
      if (xi >= 1.0) { return 1.0; }
      constexpr real_t pi = 3.1415926535897932384626433832795;
      return std::pow(std::sin(0.5 * pi * xi), 2);
   }

public:
   WindowedModalEnergyObjective(
      ParFiniteElementSpace *fes,
      std::unique_ptr<VectorCoefficient> converted_mode_,
      std::unique_ptr<VectorCoefficient> residual_mode_,
      real_t converted_weight_, real_t residual_weight_,
      real_t window_start_, real_t window_ramp_, MPI_Comm comm_)
      : TimeIntegratedObjective(fes, comm_),
        converted_mode(converted_mode_.get()),
        residual_mode(residual_mode_.get()),
        owned_converted_mode(std::move(converted_mode_)),
        owned_residual_mode(std::move(residual_mode_)),
        converted_weight(converted_weight_),
        residual_weight(residual_weight_), window_start(window_start_),
        window_ramp(window_ramp_)
   {
      MFEM_VERIFY(converted_mode && residual_mode &&
                  converted_mode->GetVDim() == residual_mode->GetVDim() &&
                  std::isfinite(converted_weight) && converted_weight > 0.0 &&
                  std::isfinite(residual_weight) && residual_weight >= 0.0 &&
                  std::isfinite(window_start) && std::isfinite(window_ramp) &&
                  window_ramp >= 0.0,
                  "Windowed modal-energy objective has invalid parameters.");
   }

   real_t EvaluateInstantaneous(const ParGridFunction &u,
                                real_t time) override
   {
      const real_t window = Window(time);
      if (window == 0.0) { return 0.0; }
      const real_t converted_projection = EvaluateDisplacementModalProjection(
         *fespace, u, *converted_mode, comm);
      const real_t residual_projection = EvaluateDisplacementModalProjection(
         *fespace, u, *residual_mode, comm);
      return 0.5 * window *
             (residual_weight * residual_projection * residual_projection -
              converted_weight * converted_projection * converted_projection);
   }

   void AssembleStateGradientScaled(
      const ParGridFunction &u, real_t time, real_t scale,
      ParLinearForm &grad_form) override
   {
      const real_t window = Window(time);
      if (window == 0.0)
      {
         grad_form = 0.0;
         return;
      }
      const real_t converted_projection = EvaluateDisplacementModalProjection(
         *fespace, u, *converted_mode, comm);
      const real_t residual_projection = EvaluateDisplacementModalProjection(
         *fespace, u, *residual_mode, comm);
      const real_t converted_factor =
         -scale * window * converted_weight * converted_projection;
      const real_t residual_factor =
         scale * window * residual_weight * residual_projection;

      class ModalEnergyGradientCoefficient : public VectorCoefficient
      {
      private:
         VectorCoefficient &converted;
         VectorCoefficient &residual;
         real_t converted_factor;
         real_t residual_factor;
         Vector converted_value, residual_value;

      public:
         ModalEnergyGradientCoefficient(VectorCoefficient &converted_,
                                        VectorCoefficient &residual_,
                                        real_t converted_factor_,
                                        real_t residual_factor_)
            : VectorCoefficient(converted_.GetVDim()), converted(converted_),
              residual(residual_), converted_factor(converted_factor_),
              residual_factor(residual_factor_) {}

         void Eval(Vector &value, ElementTransformation &T,
                   const IntegrationPoint &ip) override
         {
            converted.Eval(converted_value, T, ip);
            residual.Eval(residual_value, T, ip);
            value = converted_value;
            value *= converted_factor;
            value.Add(residual_factor, residual_value);
         }
      };

      ModalEnergyGradientCoefficient gradient(
         *converted_mode, *residual_mode, converted_factor, residual_factor);

      class HighOrderVectorDomainLFIntegrator : public LinearFormIntegrator
      {
      private:
         Vector shape, value;
         VectorCoefficient &coefficient;

      public:
         explicit HighOrderVectorDomainLFIntegrator(
            VectorCoefficient &coefficient_)
            : coefficient(coefficient_) {}

         void AssembleRHSElementVect(const FiniteElement &el,
                                     ElementTransformation &T,
                                     Vector &elvect) override
         {
            const int vdim = coefficient.GetVDim();
            const int dof = el.GetDof();
            shape.SetSize(dof);
            elvect.SetSize(dof * vdim);
            elvect = 0.0;
            const IntegrationRule &ir =
               IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 2);
            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               T.SetIntPoint(&ip);
               el.CalcPhysShape(T, shape);
               coefficient.Eval(value, T, ip);
               const real_t weight = ip.weight * T.Weight();
               for (int component = 0; component < vdim; component++)
               {
                  const real_t scaled_value = weight * value(component);
                  for (int basis = 0; basis < dof; basis++)
                  {
                     elvect(dof * component + basis) +=
                        scaled_value * shape(basis);
                  }
               }
            }
         }

         using LinearFormIntegrator::AssembleRHSElementVect;
      };

      grad_form.AddDomainIntegrator(
         new HighOrderVectorDomainLFIntegrator(gradient));
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

   real_t EvaluateInstantaneous(const ParGridFunction &u,
                                real_t time) override
   {
      (void)time;
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

      return global_work;
   }

   void AssembleStateGradientScaled(
      const ParGridFunction &u, real_t time, real_t scale,
      ParLinearForm &grad_form) override
   {
      (void)u;
      (void)time;
      // ∂J/∂u = f (the applied load)

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

      ScaledLoadCoef scaled_load(applied_load, scale);

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
