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

#ifndef MFEM_CEED_ALGEBRAIC_HPP
#define MFEM_CEED_ALGEBRAIC_HPP

#include "../../config/config.hpp"

#include "../fespacehierarchy.hpp"
#include "../multigrid.hpp"
#include "util.hpp"
#include "operator.hpp"

namespace mfem
{

namespace ceed
{

#ifdef MFEM_USE_CEED

/** @brief Assembles a CeedOperator as an mfem::SparseMatrix

    In parallel, this assembles independently on each processor, that is, it
    assembles at the L-vector level. The assembly procedure is always
    performed on the host, but this works for operators stored on device
    by copying memory. */
int CeedOperatorFullAssemble(CeedOperator op, SparseMatrix **mat);

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
   virtual const mfem::Operator *GetProlongationMatrix() const override { return NULL; }
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
   virtual const mfem::Operator *GetProlongationMatrix() const override { return P; }
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

#endif // MFEM_USE_MPI

/** @brief Multigrid interpolation operator in Ceed framework

    Interpolation/restriction has two components, an element-wise
    interpolation and then a scaling to correct multiplicity
    on shared ldofs. This encapsulates those two in one object
    using the MFEM Operator interface. */
class AlgebraicInterpolation : public mfem::Operator
{
public:
   AlgebraicInterpolation(
      Ceed ceed, CeedBasis basisctof,
      CeedElemRestriction erestrictu_coarse,
      CeedElemRestriction erestrictu_fine);

   ~AlgebraicInterpolation();

   virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;

   virtual void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const;

   using mfem::Operator::SetupRAP;
private:
   int Initialize(Ceed ceed, CeedBasis basisctof,
                  CeedElemRestriction erestrictu_coarse,
                  CeedElemRestriction erestrictu_fine);
   int Finalize();

   CeedBasis basisctof_;
   CeedVector u_, v_;

   bool owns_basis_;

   CeedQFunction qf_restrict, qf_prolong;
   CeedOperator op_interp, op_restrict;
   CeedVector fine_multiplicity_r;
   CeedVector fine_work;
};

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
   void PrependPCoarsenedLevel(int current_order, int order_reduction);

private:
   CeedElemRestriction fine_er;
   Array<AlgebraicInterpolation*> ceed_interpolations;
   Array<TransposeOperator*> R_tr;
};

/** @brief Extension of Multigrid object to algebraically generated coarse spaces */
class AlgebraicMultigrid : public GeometricMultigrid
{
public:
   /** @brief Constructs multigrid solver based on existing space hierarchy

       This only works if the Ceed device backend is enabled.

       @param[in] hierarchy  Hierarchy of (algebraic) spaces
       @param[in] form       partially assembled BilinearForm on finest level
       @param[in] ess_tdofs  List of essential true dofs on finest level
    */
   AlgebraicMultigrid(
      AlgebraicSpaceHierarchy &hierarchy,
      BilinearForm &form,
      const Array<int> &ess_tdofs
   );
   virtual void SetOperator(const mfem::Operator &op) override { }
   ~AlgebraicMultigrid();

private:
   OperatorHandle fine_operator;
   Array<CeedOperator> ceed_operators;
};
#endif // MFEM_USE_CEED

/** @brief Wrapper for AlgebraicMultigrid object

    This exists so that the algebraic Ceed-based idea has the simplest
    possible one-line interface. Finer control (choosing smoothers, w-cycle)
    can be exercised with the AlgebraicMultigrid object. */
class AlgebraicSolver : public Solver
{
private:
#ifdef MFEM_USE_CEED
   AlgebraicSpaceHierarchy * fespaces;
   AlgebraicMultigrid * multigrid;
#endif

public:
   /** @brief Constructs algebraic multigrid hierarchy and solver.

       This only works if the Ceed device backend is enabled.

       @param[in] form      partially assembled BilinearForm on finest level
       @param[in] ess_tdofs List of essential true dofs on finest level
    */
   AlgebraicSolver(BilinearForm &form, const Array<int>& ess_tdofs);
   ~AlgebraicSolver();
   void Mult(const Vector& x, Vector& y) const;
   void SetOperator(const mfem::Operator& op);
};

} // namespace ceed

} // namespace mfem



#endif // MFEM_CEED_ALGEBRAIC_HPP
