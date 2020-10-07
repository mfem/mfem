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

#ifndef MFEM_CEED_ALGEBRAIC_HPP
#define MFEM_CEED_ALGEBRAIC_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_CEED
#include "../fem/fespacehierarchy.hpp"
#include "../fem/multigrid.hpp"
#include "../fem/libceed/ceedsolvers-utility.h"

namespace mfem
{

// forward declarations
class CeedMultigridLevel;
class BilinearForm;

class AlgebraicCoarseFESpace : public FiniteElementSpace
{
public:
   AlgebraicCoarseFESpace(
      FiniteElementSpace &fine_fes,
      CeedElemRestriction fine_er,
      int order,
      int dim,
      int order_reduction_
   );
   int GetOrderReduction() const { return order_reduction; }
   CeedElemRestriction GetCeedElemRestriction() const { return ceed_elem_restriction; }
   CeedBasis GetCeedCoarseToFine() const { return coarse_to_fine; }
   virtual const Operator *GetProlongationMatrix() const override { return NULL; }
private:
   int order_reduction;
   CeedElemRestriction ceed_elem_restriction;
   CeedBasis coarse_to_fine;
};

// class ParAlgebraicCoarseFESpace : public AlgebraicCoarseFESpace, public ParFiniteElementSpace
// {
// };

class AlgebraicFESpaceHierarchy : public FiniteElementSpaceHierarchy
{
public:
   AlgebraicFESpaceHierarchy(FiniteElementSpace &fespace);
   AlgebraicCoarseFESpace& GetAlgebraicCoarseFESpace(int level)
   {
      MFEM_ASSERT(level < GetNumLevels() - 1, "");
      return static_cast<AlgebraicCoarseFESpace&>(*fespaces[level]);
   }
private:
   // TODO: delete these
   CeedElemRestriction fine_er;
   CeedBasis fine_basis;
};

// class ParAlgebraicFESpaceHierarchy : public AlgebraicFESpaceHierarchy, public ParFiniteElementSpaceHierarchy
// {

// };

class AlgebraicCeedMultigrid : public Multigrid
{
public:
   AlgebraicCeedMultigrid(
      AlgebraicFESpaceHierarchy &hierarchy,
      BilinearForm &form,
      Array<int> ess_tdofs
   );
private:
   OperatorHandle fine_operator;
   Array<CeedOperator> ceed_operators;
};

class AlgebraicCeedSolver : public mfem::Solver
{
public:
   AlgebraicCeedSolver(Operator& fine_mfem_op, BilinearForm& form,
                       Array<int>& ess_dofs, bool use_amg=false);
   ~AlgebraicCeedSolver();

   /// Note that this does not rebuild the hierarchy or smoothers,
   /// just changes the finest level operator for residual computations
   void SetOperator(const Operator& op) { operators[0] = const_cast<Operator*>(&op); }

   void Mult(const Vector& x, Vector& y) const;

private:
   int num_levels;
   Operator ** operators;
   CeedMultigridLevel ** levels;
   Solver ** solvers;
   CeedOperator fine_composite_op;
};

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_CEED_ALGEBRAIC_HPP
