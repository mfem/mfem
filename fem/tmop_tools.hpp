// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
   const double dt_scale;
   const AssemblyLevel al;
   MemoryType opt_mt = MemoryType::DEFAULT;

   void ComputeAtNewPositionScalar(const Vector &new_nodes, Vector &new_field);
public:
   AdvectorCG(AssemblyLevel al = AssemblyLevel::LEGACY,
              double timestep_scale = 0.5)
      : AdaptivityEvaluator(),
        ode_solver(), nodes0(), field0(), dt_scale(timestep_scale), al(al) { }

   virtual void SetInitialField(const Vector &init_nodes,
                                const Vector &init_field);

   virtual void ComputeAtNewPosition(const Vector &new_nodes,
                                     Vector &new_field,
                                     int new_nodes_ordering = Ordering::byNODES);

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
public:
   InterpolatorFP() : finder(NULL) { }

   virtual void SetInitialField(const Vector &init_nodes,
                                const Vector &init_field);

   virtual void ComputeAtNewPosition(const Vector &new_nodes,
                                     Vector &new_field,
                                     int new_nodes_ordering = Ordering::byNODES);

   const FindPointsGSLIB *GetFindPointsGSLIB() const
   {
      return finder;
   }

   ~InterpolatorFP()
   {
      finder->FreeData();
      delete finder;
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

   virtual void Mult(const Vector &ind, Vector &di_dt) const;
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

   virtual void Mult(const Vector &ind, Vector &di_dt) const;
};
#endif

class TMOPNewtonSolver : public LBFGSSolver
{
protected:
   // 0 - Newton, 1 - LBFGS.
   int solver_type;
   bool parallel;

   // Line search step is rejected if min(detJ) <= min_detJ_threshold.
   double min_detJ_threshold = 0.0;

   // Surface fitting variables.
   mutable double surf_fit_err_avg_prvs = 10000.0;
   mutable double surf_fit_err_avg, surf_fit_err_max;
   mutable bool update_surf_fit_coeff = false;
   double surf_fit_max_threshold = -1.0;
   double surf_fit_rel_change_threshold = 0.001;
   double surf_fit_scale_factor = 0.0;
   mutable int adapt_inc_count = 0;
   mutable int max_adapt_inc_count = 10;

   // Minimum determinant over the whole mesh. Used for mesh untangling.
   double *min_det_ptr = nullptr;
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

   void UpdateDiscreteTC(const TMOP_Integrator &ti, const Vector &x_new,
                         int x_ordering = Ordering::byNODES) const;

   double ComputeMinDet(const Vector &x_loc,
                        const FiniteElementSpace &fes) const;

   double MinDetJpr_2D(const FiniteElementSpace*, const Vector&) const;
   double MinDetJpr_3D(const FiniteElementSpace*, const Vector&) const;

   /** @name Methods for adaptive surface fitting weight. */
   ///@{
   /// Get the average and maximum surface fitting error at the marked nodes.
   /// If there is more than 1 TMOP integrator, we get the maximum of the
   /// average and maximum error over all integrators.
   virtual void GetSurfaceFittingError(double &err_avg, double &err_max) const;

   /// Update surface fitting weight as surf_fit_weight *= factor.
   void UpdateSurfaceFittingWeight(double factor) const;

   /// Get the surface fitting weight for all the TMOP integrators.
   void GetSurfaceFittingWeight(Array<double> &weights) const;
   ///@}

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

   void SetMinDetPtr(double *md_ptr) { min_det_ptr = md_ptr; }

   /// Set the memory type for temporary memory allocations.
   void SetTempMemoryType(MemoryType mt) { temp_mt = mt; }

   /// Compute scaling factor for the node movement direction using line-search.
   /// We impose constraints on TMOP energy, gradient, minimum Jacobian of
   /// the mesh, and (optionally) on the surface fitting error.
   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;

   /// Update (i) discrete functions at new nodal positions, and
   /// (ii) surface fitting weight.
   virtual void ProcessNewState(const Vector &x) const;

   /** @name Methods for adaptive surface fitting weight. (Experimental) */
   /// Enable/Disable adaptive surface fitting weight.
   /// The weight is modified after each TMOPNewtonSolver iteration as:
   /// w_{k+1} = w_{k} * @a surf_fit_scale_factor if relative change in
   /// max surface fitting error < @a surf_fit_rel_change_threshold.
   /// The solver terminates if the maximum surface fitting error does
   /// not sufficiently decrease for @a max_adapt_inc_count consecutive
   /// solver iterations or if the max error falls below @a surf_fit_max_threshold.
   void EnableAdaptiveSurfaceFitting()
   {
      surf_fit_scale_factor = 10.0;
      surf_fit_rel_change_threshold = 0.001;
   }
   void SetAdaptiveSurfaceFittingScalingFactor(double factor)
   {
      surf_fit_scale_factor = factor;
   }
   void SetAdaptiveSurfaceFittingRelativeChangeThreshold(double threshold)
   {
      surf_fit_rel_change_threshold = threshold;
   }
   void SetMaxNumberofIncrementsForAdaptiveFitting(int count)
   {
      max_adapt_inc_count = count;
   }
   void SetTerminationWithMaxSurfaceFittingError(double max_error)
   {
      surf_fit_max_threshold = max_error;
   }
   void SetMinimumDeterminantThreshold(double threshold)
   {
      min_detJ_threshold = threshold;
   }

   virtual void Mult(const Vector &b, Vector &x) const
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

   virtual void SetSolver(Solver &solver)
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
   virtual void SetPreconditioner(Solver &pr) { SetSolver(pr); }
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
