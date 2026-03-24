// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PNONLINEARFORM
#define MFEM_PNONLINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pgridfunc.hpp"
#include "nonlinearform.hpp"

namespace mfem
{

/// Parallel non-linear operator on the true dofs
class ParNonlinearForm : public NonlinearForm
{
protected:
   mutable ParGridFunction X, Y;
   mutable OperatorHandle pGrad;

   void GradientSharedFaces(const Vector &x, int skip_zeros = 1) const;

public:
   ParNonlinearForm(ParFiniteElementSpace *pf);

   ParFiniteElementSpace *ParFESpace() const
   { return (ParFiniteElementSpace *)fes; }

   /// Compute the energy corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a "GridFunction size" vector, i.e. its size must
       be fes->GetVSize(). */
   real_t GetParGridFunctionEnergy(const Vector &x) const;

   /// Compute the energy of a ParGridFunction
   real_t GetEnergy(const ParGridFunction &x) const
   { return GetParGridFunctionEnergy(x); }

   real_t GetEnergy(const Vector &x) const override
   { return GetParGridFunctionEnergy(Prolongate(x)); }

   void Mult(const Vector &x, Vector &y) const override;

   /// Return the local gradient matrix for the given true-dof vector x.
   /** The returned matrix does NOT have any boundary conditions imposed. */
   const SparseMatrix &GetLocalGradient(const Vector &x) const;

   Operator &GetGradient(const Vector &x) const override;

   /// Set the operator type id for the parallel gradient matrix/operator.
   void SetGradientType(Operator::Type tid) { pGrad.SetType(tid); }

   /** @brief Update the ParNonlinearForm to propagate updates of the associated
       parallel FE space. */
   /** After calling this method, the essential boundary conditions need to be
       set again. */
   void Update() override;

   virtual ~ParNonlinearForm() { }
};


/** @brief A class representing a general parallel block nonlinear operator
    defined on the Cartesian product of multiple ParFiniteElementSpace%s. */
/** The ParBlockNonlinearForm takes as input, and returns as output, vectors on
    the true dofs. */
class ParBlockNonlinearForm : public BlockNonlinearForm
{
protected:
   mutable BlockVector xs_true, ys_true;
   mutable Array2D<OperatorHandle *> phBlockGrad;
   mutable BlockOperator *pBlockGrad;

   void GradientSharedFaces(const BlockVector &xs, int skip_zeros) const;

public:
   /// Computes the energy of the system
   real_t GetEnergy(const Vector &x) const override;

   /// Construct an empty ParBlockNonlinearForm. Initialize with SetParSpaces().
   ParBlockNonlinearForm() : pBlockGrad(NULL) { }

   /** @brief Construct a ParBlockNonlinearForm on the given set of
       ParFiniteElementSpace%s. */
   ParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf);

   /// Return the @a k-th parallel FE space of the ParBlockNonlinearForm.
   ParFiniteElementSpace *ParFESpace(int k);
   /** @brief Return the @a k-th parallel FE space of the ParBlockNonlinearForm
       (const version). */
   const ParFiniteElementSpace *ParFESpace(int k) const;

   /** @brief After a call to SetParSpaces(), the essential b.c. and the
       gradient-type (if different from the default) must be set again. */
   void SetParSpaces(Array<ParFiniteElementSpace *> &pf);

   /** @brief Set essential boundary conditions to each finite element space
       using boundary attribute markers.

       This method calls `FiniteElementSpace::GetEssentialTrueDofs()` for each
       space and stores ess_tdof_lists internally.

       If `rhs` vectors are non-null, the entries corresponding to these
       essential DoFs are set to zero. This ensures compatibility with the
       output of the `Mult()` method, which also zeroes out these entries.

       @param[in] bdr_attr_is_ess A list of boundary attribute markers for each
       space.
       @param[in,out] rhs         An array of optional right-hand side vectors.
       If a vector at `rhs[i]` is non-null, its essential DoFs will be set
       to zero. */
   virtual void SetEssentialBC(const Array<Array<int>*> &bdr_attr_is_ess,
                               Array<Vector*> &rhs) override;

   /** @brief Set essential boundary conditions to each finite element space
       using essential true dof lists.

       This method stores a copy of the provided essential true dof lists.

       If `rhs` vectors are non-null, the entries corresponding to these
       essential DoFs are set to zero. This ensures compatibility with the
       output of the `Mult()` method, which also zeroes out these entries.

       @param[in] ess_tdof_list A list of essential true dofs for each space.
       @param[in,out] rhs       An array of optional right-hand side vectors.
       If a vector at `rhs[i]` is non-null, its essential DoFs will be set
       to zero. */
   virtual void SetEssentialTrueDofs(const Array<Array<int>*> &ess_tdof_list,
                                     Array<Vector*> &rhs) override;

   /// Block T-Vector to Block T-Vector
   void Mult(const Vector &x, Vector &y) const override;

   /// Return the local block gradient matrix for the given true-dof vector x
   const BlockOperator &GetLocalGradient(const Vector &x) const;

   BlockOperator &GetGradient(const Vector &x) const override;

   /** @brief Set the operator type id for the blocks of the parallel gradient
       matrix/operator. The default type is Operator::Hypre_ParCSR. */
   void SetGradientType(Operator::Type tid);

   /// Destructor.
   virtual ~ParBlockNonlinearForm();
};

}

#endif // MFEM_USE_MPI

#endif
