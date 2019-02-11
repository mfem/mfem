// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TMOP_TOOLS_HPP
#define MFEM_TMOP_TOOLS_HPP

#include "../fem/pbilinearform.hpp"
#include "../fem/tmop.hpp"

namespace mfem
{

// Performs an advection step.
class AdvectorCGOperator : public TimeDependentOperator
{
private:
   const Vector &x0;
   Vector &x_now;
   ParGridFunction &u;

   VectorGridFunctionCoefficient u_coeff;
   mutable ParBilinearForm M, K;

public:
   // Note: pfes must be the ParFESpace of the mesh that will be moved.
   //       xn must be the Nodes valus of the mesh that will be moved.
   AdvectorCGOperator(const Vector &x_start, ParGridFunction &vel,
                      Vector &xn, ParFiniteElementSpace &pfes);

   virtual void Mult(const Vector &ind, Vector &di_dt) const;
};

// Performs the whole advection loop.
class AdvectorCG : public ParAdaptivityEvaluator
{
private:
   RK4Solver ode_solver;

public:
   AdvectorCG() : ode_solver() { }

   virtual void ComputeAtNewPosition(const Vector &start_nodes,
                                     const Vector &new_nodes,
                                     Vector &field);
};

}

#endif
