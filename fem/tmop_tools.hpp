// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
                                     Vector &new_field);

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
   int dim;
public:
   InterpolatorFP() : finder(NULL) { }

   virtual void SetInitialField(const Vector &init_nodes,
                                const Vector &init_field);

   virtual void ComputeAtNewPosition(const Vector &new_nodes,
                                     Vector &new_field);

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

   // Minimum determinant over the whole mesh. Used for mesh untangling.
   double *min_det_ptr = nullptr;

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

   void UpdateDiscreteTC(const TMOP_Integrator &ti, const Vector &x_new) const;

   double ComputeMinDet(const Vector &x_loc,
                        const FiniteElementSpace &fes) const;

   double MinDetJpr_2D(const FiniteElementSpace*, const Vector&) const;
   double MinDetJpr_3D(const FiniteElementSpace*, const Vector&) const;

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

   // Set the memory type for temporary memory allocations.
   void SetTempMemoryType(MemoryType mt) { temp_mt = mt; }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;

   virtual void ProcessNewState(const Vector &x) const;

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
