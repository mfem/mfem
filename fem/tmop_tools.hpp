// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

namespace mfem
{

// Performs the full remap advection loop.
class AdvectorCG : public AdaptivityEvaluator
{
private:
   RK4Solver ode_solver;
   Vector nodes0;
   Vector field0;

public:
   AdvectorCG() : AdaptivityEvaluator(), ode_solver(), nodes0(), field0() { }

   virtual void SetInitialField(const Vector &init_nodes,
                                const Vector &init_field);

   virtual void ComputeAtNewPosition(const Vector &new_nodes,
                                     Vector &new_field);
};

/// Performs a single remap advection step in serial.
class SerialAdvectorCGOper : public TimeDependentOperator
{
protected:
   const Vector &x0;
   Vector &x_now;
   GridFunction &u;
   VectorGridFunctionCoefficient u_coeff;
   mutable BilinearForm M, K;

public:
   /** Here @a fes is the FESpace of the function that will be moved. Note
       that Mult() moves the nodes of the mesh corresponding to @a fes. */
   SerialAdvectorCGOper(const Vector &x_start, GridFunction &vel,
                        FiniteElementSpace &fes);

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

public:
   /** Here @a pfes is the ParFESpace of the function that will be moved. Note
       that Mult() moves the nodes of the mesh corresponding to @a pfes. */
   ParAdvectorCGOper(const Vector &x_start, GridFunction &vel,
                     ParFiniteElementSpace &pfes);

   virtual void Mult(const Vector &ind, Vector &di_dt) const;
};
#endif

class TMOPNewtonSolver : public NewtonSolver
{
private:
   bool parallel;

   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;

   mutable DiscreteAdaptTC *discr_tc;

public:
#ifdef MFEM_USE_MPI
   TMOPNewtonSolver(MPI_Comm comm, const IntegrationRule &irule)
      : NewtonSolver(comm), parallel(true), ir(irule), discr_tc(NULL) { }
#endif
   TMOPNewtonSolver(const IntegrationRule &irule)
      : NewtonSolver(), parallel(false), ir(irule), discr_tc(NULL) { }

   void SetDiscreteAdaptTC(DiscreteAdaptTC *tc) { discr_tc = tc; }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;

   virtual void ProcessNewState(const Vector &x) const;
};

/// Allows negative Jacobians. Used for untangling.
class TMOPDescentNewtonSolver : public NewtonSolver
{
private:
   bool parallel;

   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;

   mutable DiscreteAdaptTC *discr_tc;

public:
#ifdef MFEM_USE_MPI
   TMOPDescentNewtonSolver(MPI_Comm comm, const IntegrationRule &irule)
      : NewtonSolver(comm), parallel(true), ir(irule), discr_tc(NULL) { }
#endif
   TMOPDescentNewtonSolver(const IntegrationRule &irule)
      : NewtonSolver(), parallel(false), ir(irule), discr_tc(NULL) { }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const;

   virtual void ProcessNewState(const Vector &x) const;
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
