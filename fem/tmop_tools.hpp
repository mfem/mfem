// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TMOP_TOOLS_HPP
#define MFEM_TMOP_TOOLS_HPP

#include "bilinearform.hpp"
#include "pbilinearform.hpp"
#include "tmop.hpp"
#include "gslib.hpp"

namespace mfem
{

// Performs the full remap advection loop.
class AdvectorCG : public AdaptivityEvaluator
{
private:
   RK4Solver ode_solver;
   Vector nodes0;
   Vector field0;
   const real_t dt_scale;
   const AssemblyLevel al;
   MemoryType opt_mt = MemoryType::DEFAULT;

   void ComputeAtNewPositionScalar(const Vector &new_nodes, Vector &new_field);
public:
   AdvectorCG(AssemblyLevel al = AssemblyLevel::LEGACY,
              real_t timestep_scale = 0.5)
      : AdaptivityEvaluator(),
        ode_solver(), nodes0(), field0(), dt_scale(timestep_scale), al(al) { }

   void SetInitialField(const Vector &init_nodes,
                        const Vector &init_field) override;

   void ComputeAtNewPosition(const Vector &new_nodes,
                             Vector &new_field,
                             int new_nodes_ordering = Ordering::byNODES) override;

   /// Set the memory type used for large memory allocations. This memory type
   /// is used when constructing the AdvectorCGOper but currently only for the
   /// parallel variant.
   void SetMemoryType(MemoryType mt) { opt_mt = mt; }
};

#ifdef MFEM_USE_GSLIB
class InterpolatorFP : public AdaptivityEvaluator
{
private:
   Vector nodes0;
   GridFunction field0_gf;
   FindPointsGSLIB *finder;
   // FE space for the nodes of the solution GridFunction.
   FiniteElementSpace *fes_field_nodes;

   void GetFieldNodesPosition(const Vector &mesh_nodes,
                              Vector &nodes_pos) const;

public:
   InterpolatorFP() : finder(NULL), fes_field_nodes(NULL) { }

   void SetInitialField(const Vector &init_nodes,
                        const Vector &init_field) override;

   void ComputeAtNewPosition(const Vector &new_nodes,
                             Vector &new_field,
                             int new_nodes_ordering = Ordering::byNODES) override;

   const FindPointsGSLIB *GetFindPointsGSLIB() const
   {
      return finder;
   }

   ~InterpolatorFP()
   {
      finder->FreeData();
      delete finder;
      delete fes_field_nodes;
   }
};
#endif

/// Performs a single remap advection step in serial.
class SerialAdvectorCGOper : public TimeDependentOperator
{
protected:
   const Vector &x0;
   Vector &x_now;
   GridFunction &u;
   VectorGridFunctionCoefficient u_coeff;
   mutable BilinearForm M, K;
   const AssemblyLevel al;

public:
   /** Here @a fes is the FESpace of the function that will be moved. Note
       that Mult() moves the nodes of the mesh corresponding to @a fes. */
   SerialAdvectorCGOper(const Vector &x_start, GridFunction &vel,
                        FiniteElementSpace &fes,
                        AssemblyLevel al = AssemblyLevel::LEGACY);

   void Mult(const Vector &ind, Vector &di_dt) const override;
};

#ifdef MFEM_USE_MPI
/// Performs a single remap advection step in parallel.
class ParAdvectorCGOper : public TimeDependentOperator
{
protected:
   const Vector &x0;
   Vector &x_now;
   GridFunction &u;
   VectorGridFunctionCoefficient u_coeff;
   mutable ParBilinearForm M, K;
   const AssemblyLevel al;

public:
   /** Here @a pfes is the ParFESpace of the function that will be moved. Note
       that Mult() moves the nodes of the mesh corresponding to @a pfes.
       @a mt is used to set the memory type of the integrators. */
   ParAdvectorCGOper(const Vector &x_start, GridFunction &vel,
                     ParFiniteElementSpace &pfes,
                     AssemblyLevel al = AssemblyLevel::LEGACY,
                     MemoryType mt = MemoryType::DEFAULT);

   void Mult(const Vector &ind, Vector &di_dt) const override;
};
#endif

class TMOPNewtonSolver : public LBFGSSolver
{
protected:
   // 0 - Newton, 1 - LBFGS.
   int solver_type;
   bool parallel;

   // Line search step is rejected if min(detJ) <= min_detJ_limit.
   real_t min_detJ_limit = 0.0;

   // Surface fitting variables.
   mutable real_t surf_fit_avg_err_prvs = 10000.0;
   mutable real_t surf_fit_avg_err, surf_fit_max_err;
   mutable bool surf_fit_coeff_update = false;
   real_t surf_fit_max_err_limit = -1.0;
   real_t surf_fit_err_rel_change_limit = 0.001;
   real_t surf_fit_scale_factor = 0.0;
   mutable int surf_fit_adapt_count = 0;
   mutable int surf_fit_adapt_count_limit = 10;
   mutable real_t surf_fit_weight_limit = 1e10;
   bool surf_fit_converge_error = false;

   // Minimum determinant over the whole mesh. Used for mesh untangling.
   real_t *min_det_ptr = nullptr;
   // Flag to compute minimum determinant and maximum metric in ProcessNewState,
   // which is required for TMOP_WorstCaseUntangleOptimizer_Metric.
   mutable bool compute_metric_quantile_flag = true;

   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;
   // These fields are relevant for mixed meshes.
   IntegrationRules *IntegRules;
   int integ_order;

   MemoryType temp_mt = MemoryType::DEFAULT;

   const IntegrationRule &GetIntegrationRule(const FiniteElement &el) const
   {
      if (IntegRules)
      {
         return IntegRules->Get(el.GetGeomType(), integ_order);
      }
      return ir;
   }

   real_t ComputeMinDet(const Vector &x_loc,
                        const FiniteElementSpace &fes) const;

   real_t MinDetJpr_2D(const FiniteElementSpace*, const Vector&) const;
   real_t MinDetJpr_3D(const FiniteElementSpace*, const Vector&) const;

   /** @name Methods for adaptive surface fitting weight. */
   ///@{
   /// Get the average and maximum surface fitting error at the marked nodes.
   /// If there is more than 1 TMOP integrator, we get the maximum of the
   /// average and maximum error over all integrators.
   virtual void GetSurfaceFittingError(const Vector &x_loc,
                                       real_t &err_avg, real_t &err_max) const;

   /// Update surface fitting weight as surf_fit_weight *= factor.
   void UpdateSurfaceFittingWeight(real_t factor) const;

   /// Get the surface fitting weight for all the TMOP integrators.
   void GetSurfaceFittingWeight(Array<real_t> &weights) const;
   ///@}

   /// Check if surface fitting is enabled.
   bool IsSurfaceFittingEnabled() const;

public:
#ifdef MFEM_USE_MPI
   TMOPNewtonSolver(MPI_Comm comm, const IntegrationRule &irule, int type = 0)
      : LBFGSSolver(comm), solver_type(type), parallel(true),
        ir(irule), IntegRules(NULL), integ_order(-1) { }
#endif
   TMOPNewtonSolver(const IntegrationRule &irule, int type = 0)
      : LBFGSSolver(), solver_type(type), parallel(false),
        ir(irule), IntegRules(NULL), integ_order(-1) { }

   /// Prescribe a set of integration rules; relevant for mixed meshes.
   /** If called, this function has priority over the IntegrationRule given to
       the constructor of the class. */
   void SetIntegrationRules(IntegrationRules &irules, int order)
   {
      IntegRules = &irules;
      integ_order = order;
   }

   void SetMinDetPtr(real_t *md_ptr) { min_det_ptr = md_ptr; }

   /// Set the memory type for temporary memory allocations.
   void SetTempMemoryType(MemoryType mt) { temp_mt = mt; }

   /// Compute scaling factor for the node movement direction using line-search.
   /// We impose constraints on TMOP energy, gradient, minimum Jacobian of
   /// the mesh, and (optionally) on the surface fitting error.
   real_t ComputeScalingFactor(const Vector &x, const Vector &b) const override;

   /// Update (i) discrete functions at new nodal positions, and
   /// (ii) surface fitting weight.
   void ProcessNewState(const Vector &x) const override;

   /** @name Methods for adaptive surface fitting.
       \brief These methods control the behavior of the weight and the
       termination of the solver. (Experimental)

       Adaptive fitting weight: The weight is modified after each
       TMOPNewtonSolver iteration as:
       w_{k+1} = w_{k} * \ref surf_fit_scale_factor if the relative
       change in average fitting error < \ref surf_fit_err_rel_change_limit.
       When converging based on the residual, we enforce the fitting weight
       to be at-most \ref surf_fit_weight_limit, and increase it only if the
       fitting error is below user prescribed threshold
       (\ref surf_fit_max_err_limit).
       See \ref SetAdaptiveSurfaceFittingScalingFactor and
       \ref SetAdaptiveSurfaceFittingRelativeChangeThreshold.

       Note that the solver stops if the maximum surface fitting error
       does not sufficiently decrease for \ref surf_fit_adapt_count_limit (default 10)
       consecutive increments of the fitting weight during weight adaptation.
       This typically occurs when the mesh cannot align with the level-set
       without degrading element quality.
       See \ref SetMaxNumberofIncrementsForAdaptiveFitting.

       Convergence criterion: There are two modes, residual- and error-based,
       which can be toggled using \ref SetSurfaceFittingConvergenceBasedOnError.

       (i) Residual based (default): Stop when the norm of the gradient of the
       TMOP objective reaches the prescribed tolerance. This method is best used
       with a reasonable value for \ref surf_fit_weight_limit when the
       adaptive surface fitting scheme is used. See method
       \ref SetSurfaceFittingWeightLimit.

       (ii) Error based: Stop when the maximum fitting error
       reaches the user-prescribed threshold, \ref surf_fit_max_err_limit.
       In this case, \ref surf_fit_weight_limit is ignored during weight
       adaptation.
   */
   ///@{
   void SetAdaptiveSurfaceFittingScalingFactor(real_t factor)
   {
      MFEM_VERIFY(factor > 1.0, "Scaling factor must be greater than 1.");
      surf_fit_scale_factor = factor;
   }
   void SetAdaptiveSurfaceFittingRelativeChangeThreshold(real_t threshold)
   {
      surf_fit_err_rel_change_limit = threshold;
   }
   /// Used for stopping based on the number of consecutive failed weight
   /// adaptation iterations.
   // TODO: Rename to SetMaxNumberofIncrementsForAdaptiveSurfaceFitting
   // in future.
   void SetMaxNumberofIncrementsForAdaptiveFitting(int count)
   {
      surf_fit_adapt_count_limit = count;
   }
   /// Used for error-based surface fitting termination.
   void SetTerminationWithMaxSurfaceFittingError(real_t max_error)
   {
      surf_fit_max_err_limit = max_error;
      surf_fit_converge_error = true;
   }
   /// Could be used with both error-based or residual-based convergence.
   void SetSurfaceFittingMaxErrorLimit(real_t max_error)
   {
      surf_fit_max_err_limit = max_error;
   }
   /// Used for residual-based surface fitting termination.
   void SetSurfaceFittingWeightLimit(real_t weight)
   {
      surf_fit_weight_limit = weight;
   }
   /// Toggle convergence based on residual or error.
   void SetSurfaceFittingConvergenceBasedOnError(bool mode)
   {
      surf_fit_converge_error = mode;
      if (surf_fit_converge_error)
      {
         MFEM_VERIFY(surf_fit_max_err_limit >= 0,
                     "Fitting error based convergence requires the user to "
                     "first set the error threshold."
                     "See SetTerminationWithMaxSurfaceFittingError");
      }
   }
   ///@}

   /// Set minimum determinant enforced during line-search.
   void SetMinimumDeterminantThreshold(real_t threshold)
   {
      min_detJ_limit = threshold;
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      if (solver_type == 0)
      {
         NewtonSolver::Mult(b, x);
      }
      else if (solver_type == 1)
      {
         LBFGSSolver::Mult(b, x);
      }
      else { MFEM_ABORT("Invalid type"); }
   }

   void SetSolver(Solver &solver) override
   {
      if (solver_type == 0)
      {
         NewtonSolver::SetSolver(solver);
      }
      else if (solver_type == 1)
      {
         LBFGSSolver::SetSolver(solver);
      }
      else { MFEM_ABORT("Invalid type"); }
   }
   void SetPreconditioner(Solver &pr) override { SetSolver(pr); }
};

void vis_tmop_metric_s(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, Mesh &pmesh,
                       char *title, int position);
#ifdef MFEM_USE_MPI
void vis_tmop_metric_p(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, ParMesh &pmesh,
                       char *title, int position);
#endif

}

#endif
