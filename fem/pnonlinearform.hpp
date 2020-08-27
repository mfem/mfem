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

public:
   ParNonlinearForm(ParFiniteElementSpace *pf);

   ParFiniteElementSpace *ParFESpace() const
   { return (ParFiniteElementSpace *)fes; }

   /// Compute the energy corresponding to the state @a x.
   /** In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a "GridFunction size" vector, i.e. its size must
       be fes->GetVSize(). */
   double GetParGridFunctionEnergy(const Vector &x) const;

   /// Compute the energy of a ParGridFunction
   double GetEnergy(const ParGridFunction &x) const
   { return GetParGridFunctionEnergy(x); }

   virtual double GetEnergy(const Vector &x) const
   { return GetParGridFunctionEnergy(Prolongate(x)); }

   virtual void Mult(const Vector &x, Vector &y) const;

   /// Return the local gradient matrix for the given true-dof vector x.
   /** The returned matrix does NOT have any boundary conditions imposed. */
   const SparseMatrix &GetLocalGradient(const Vector &x) const;

   virtual Operator &GetGradient(const Vector &x) const;

   /// Set the operator type id for the parallel gradient matrix/operator.
   void SetGradientType(Operator::Type tid) { pGrad.SetType(tid); }

   /** @brief Update the ParNonlinearForm to propagate updates of the associated
       parallel FE space. */
   /** After calling this method, the essential boundary conditions need to be
       set again. */
   virtual void Update();

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

public:
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

   // Here, rhs is a true dof vector
   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   virtual void Mult(const Vector &x, Vector &y) const;

   /// Return the local block gradient matrix for the given true-dof vector x
   const BlockOperator &GetLocalGradient(const Vector &x) const;

   virtual BlockOperator &GetGradient(const Vector &x) const;

   /** @brief Set the operator type id for the blocks of the parallel gradient
       matrix/operator. The default type is Operator::Hypre_ParCSR. */
   void SetGradientType(Operator::Type tid);

   /// Destructor.
   virtual ~ParBlockNonlinearForm();
};

}

#endif // MFEM_USE_MPI

#endif
