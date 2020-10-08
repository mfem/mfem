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
#include "../fem/libceed/ceed-wrappers.hpp"

namespace mfem
{

class AlgebraicCoarseSpace : public FiniteElementSpace
{
public:
   AlgebraicCoarseSpace(
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
   virtual const SparseMatrix *GetRestrictionMatrix() const override { return NULL; }
   ~AlgebraicCoarseSpace();
protected:
   int *dof_map;
   int order_reduction;
   CeedElemRestriction ceed_elem_restriction;
   CeedBasis coarse_to_fine;
};

#ifdef MFEM_USE_MPI

class ParAlgebraicCoarseSpace : public AlgebraicCoarseSpace
{
public:
   ParAlgebraicCoarseSpace(
      FiniteElementSpace &fine_fes,
      CeedElemRestriction fine_er,
      int order,
      int dim,
      int order_reduction_,
      GroupCommunicator *gc_fine
   );
   virtual const Operator *GetProlongationMatrix() const override { return P; }
   virtual const SparseMatrix *GetRestrictionMatrix() const override { return R_mat; }
   GroupCommunicator *GetGroupCommunicator() const { return gc; }
   HypreParMatrix *GetProlongationHypreParMatrix();
   ~ParAlgebraicCoarseSpace();
private:
   SparseMatrix *R_mat;
   GroupCommunicator *gc;
   ConformingProlongationOperator *P;
   HypreParMatrix *P_mat;
   Array<int> ldof_group, ldof_ltdof;
};

#endif

class AlgebraicSpaceHierarchy : public FiniteElementSpaceHierarchy
{
public:
   AlgebraicSpaceHierarchy(FiniteElementSpace &fespace);
   AlgebraicCoarseSpace& GetAlgebraicCoarseSpace(int level)
   {
      MFEM_ASSERT(level < GetNumLevels() - 1, "");
      return static_cast<AlgebraicCoarseSpace&>(*fespaces[level]);
   }
   ~AlgebraicSpaceHierarchy()
   {
      CeedElemRestrictionDestroy(&fine_er);
      for (int i=0; i<R_tr.Size(); ++i)
      {
         delete R_tr[i];
      }
      for (int i=0; i<ceed_interpolations.Size(); ++i)
      {
         delete ceed_interpolations[i];
      }
   }
private:
   CeedElemRestriction fine_er;
   Array<MFEMCeedInterpolation*> ceed_interpolations;
   Array<TransposeOperator*> R_tr;
};

class AlgebraicCeedMultigrid : public Multigrid
{
public:
   AlgebraicCeedMultigrid(
      AlgebraicSpaceHierarchy &hierarchy,
      BilinearForm &form,
      const Array<int> &ess_tdofs
   );
   virtual void SetOperator(const Operator &op) override { }
   ~AlgebraicCeedMultigrid();
private:
   OperatorHandle fine_operator;
   Array<CeedOperator> ceed_operators;
};

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_CEED_ALGEBRAIC_HPP
