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

#ifndef MFEM_PPRMNONLINEARFORM
#define MFEM_PPRMNONLINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pgridfunc.hpp"
#include "prmnonlinearform.hpp"

namespace mfem
{

/** @brief A class representing a general parametric parallel block nonlinear operator
    defined on the Cartesian product of multiple ParFiniteElementSpace%s. */
/** The ParPrmBlockNonlinearForm takes as input, and returns as output, vectors on
    the true dofs. */
class ParPrmBlockNonlinearForm : public PrmBlockNonlinearForm
{
protected:
   mutable BlockVector xs_true, ys_true;
   mutable Array2D<OperatorHandle *> phBlockGrad;
   mutable BlockOperator *pBlockGrad;

public:
   /// Computes the energy of the system
   virtual double GetEnergy(const Vector &x) const;

   /// Construct an empty ParPrmBlockNonlinearForm. Initialize with SetParSpaces().
   ParPrmBlockNonlinearForm() : pBlockGrad(nullptr) { }

   /** @brief Construct a ParPrmBlockNonlinearForm on the given set of
       parametric and state ParFiniteElementSpace%s. */
   ParPrmBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf, Array<ParFiniteElementSpace *> &ppf );

   /// Return the @a k-th parallel FE state space of the ParPrmBlockNonlinearForm.
   ParFiniteElementSpace *ParFESpace(int k);
   /** @brief Return the @a k-th parallel FE state space of the ParPrmBlockNonlinearForm
       (const version). */
   const ParFiniteElementSpace *ParFESpace(int k) const;

   /// Return the @a k-th parallel FE parameters space of the ParPrmBlockNonlinearForm.
   ParFiniteElementSpace *ParPrmFESpace(int k);
   /** @brief Return the @a k-th parallel FE parameters space of the ParPrmBlockNonlinearForm
       (const version). */
   const ParFiniteElementSpace *ParPrmFESpace(int k) const;


   /** @brief After a call to SetParSpaces(), the essential b.c. and the
       gradient-type (if different from the default) must be set again. */
   void SetParSpaces(Array<ParFiniteElementSpace *> &pf, Array<ParFiniteElementSpace *> &pprmf);

   // Here, rhs is a true dof vector
   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                                  Array<Vector *> &rhs);

   // Here, rhs is a true dof vector
   virtual void SetPrmEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                                  Array<Vector *> &rhs);


   /// Block T-Vector to Block T-Vector
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Block T-Vector to Block T-Vector
   virtual void PrmMult(const Vector &x, Vector &y) const;

   /// Return the local block gradient matrix for the given true-dof vector x
   const BlockOperator &GetLocalGradient(const Vector &x) const;

   virtual BlockOperator &GetGradient(const Vector &x) const;

   /** @brief Set the operator type id for the blocks of the parallel gradient
       matrix/operator. The default type is Operator::Hypre_ParCSR. */
   void SetGradientType(Operator::Type tid);

   /// Destructor.
   virtual ~ParPrmBlockNonlinearForm();

};

}

#endif
