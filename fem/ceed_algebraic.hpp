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
#include "fespacehierarchy.hpp"
#include "multigrid.hpp"
#include "libceed/ceedsolvers-utility.h"
#include "libceed/ceed-wrappers.hpp"

namespace mfem
{

/** @brief A way to use algebraic levels in a Multigrid object

    This is analogous to a FiniteElementSpace but with no Mesh information,
    constructed in a semi-algebraic way. */
class AlgebraicCoarseSpace : public FiniteElementSpace
{
public:
   AlgebraicCoarseSpace(FiniteElementSpace &fine_fes, CeedElemRestriction fine_er,
                        int order, int dim, int order_reduction_);
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

/** @brief Parallel version of AlgebraicCoarseSpace

    This provides prolongation and restriction matrices for RAP-type
    parallel operators and potential explicit assembly. */
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

/** @brief Hierarchy of AlgebraicCoarseSpace objects for use in Multigrid object */
class AlgebraicSpaceHierarchy : public FiniteElementSpaceHierarchy
{
public:
   /** @brief Construct hierarchy based on finest FiniteElementSpace

       The given space is a real (geometric) space, but the coarse spaces
       are constructed semi-algebraically with no mesh information. */
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

   /** Prepend an already constructed coarse space to the hierarchy,
       managing meshes, fespaces, and other arrays appropriately.

       Analogous to FESpaceHierarchy::AddLevel() */
   void AddCoarseLevel(AlgebraicCoarseSpace* space,
                       CeedElemRestriction er);

   /// Analogous to FiniteElementSpaceHierarchy::AddOrderRefinedLevel()
   /// could probably make this happen with just order_reduction if you
   /// want to save more info in the hierarchy
   void PrependPCoarsenedLevel(int current_order,
                               int order_reduction);

private:
   CeedElemRestriction fine_er;
   Array<MFEMCeedInterpolation*> ceed_interpolations;
   Array<TransposeOperator*> R_tr;
};

/** @brief Extension of Multigrid object to algebraically generated coarse spaces */
class AlgebraicCeedMultigrid : public GeometricMultigrid
{
public:
   /** @brief Constructs multigrid solver based on existing space hierarchy

       This only works if the Ceed device backend is enabled.

       @param[in] hierarchy   Hierarchy of (algebraic) spaces
       @param[in] form        partially assembled BilinearForm on finest level
       @param[in] ess_tdofs   List of essential true dofs on finest level
       @param[in] print_level 0 is silent
       @param[in] contrast_threshold Threshold to control p-coarsening
       @param[in] switch_amg_order Controls when to switch from p-coarsening to AMG
       @param[in] sparsification controls whether the coarsest grid is "sparsified"
                                 if it is not already lowest-order
   */
   AlgebraicCeedMultigrid(
      AlgebraicSpaceHierarchy &hierarchy,
      BilinearForm &form,
      const Array<int> &ess_tdofs,
      int print_level=1,
      double contrast_threshold=1000.0,
      int switch_amg_order=2,
      bool collocate_coarse=true,
      bool sparsification=true,
      const std::string amgx_config_file=""
   );
   virtual void SetOperator(const Operator &op) override { }
   ~AlgebraicCeedMultigrid();

private:
   OperatorHandle fine_operator;
   Array<CeedOperator> ceed_operators;
};

/** @brief Wrapper for AlgebraicCeedMultigrid object

    This exists so that the algebraic Ceed-based idea has the simplest
    possible one-line interface. Finer control (choosing smoothers, w-cycle,
    parameters) can be exercised with the AlgebraicCeedMultigrid object. */
class AlgebraicCeedSolver : public Solver
{
private:
   AlgebraicSpaceHierarchy fespaces;
   AlgebraicCeedMultigrid multigrid;

public:
   /** @brief Constructs algebraic multigrid hierarchy and solver.

       This only works if the Ceed device backend is enabled.

       @param[in] form      partially assembled BilinearForm on finest level
       @param[in] ess_tdofs List of essential true dofs on finest level
   */
   AlgebraicCeedSolver(BilinearForm &form, const Array<int>& ess_tdofs) :
      fespaces(*form.FESpace()),
      multigrid(fespaces, form, ess_tdofs)
   { }

   void Mult(const Vector& x, Vector& y) const { multigrid.Mult(x, y); }
   void SetOperator(const Operator& op) { multigrid.SetOperator(op); }
};

} // namespace mfem

#endif // MFEM_USE_CEED

#endif // MFEM_CEED_ALGEBRAIC_HPP
