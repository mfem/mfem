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
#include "paramnonlinearform.hpp"

namespace mfem
{

/** @brief A class representing a general parametric parallel block nonlinear operator
    defined on the Cartesian product of multiple ParFiniteElementSpace%s. */
/** The ParParametricBNLForm takes as input, and returns as output, vectors on
    the true dofs. */
class ParParametricBNLForm : public ParametricBNLForm
{
protected:
   mutable BlockVector xs_true, ys_true;
   mutable Array2D<OperatorHandle *> phBlockGrad;
   mutable BlockOperator *pBlockGrad;

public:
   /// Computes the energy of the system
   virtual double GetEnergy(const Vector &x) const;

   /// Construct an empty ParParametricBNLForm. Initialize with SetParSpaces().
   ParParametricBNLForm() : pBlockGrad(nullptr) { }

   /** @brief Construct a ParParametricBNLForm on the given set of
       parametric and state ParFiniteElementSpace%s. */
   ParParametricBNLForm(Array<ParFiniteElementSpace *> &statef,
                        Array<ParFiniteElementSpace *> &paramf);

   /// Return the @a k-th parallel FE state space of the ParParametricBNLForm.
   ParFiniteElementSpace *ParFESpace(int k);
   /** @brief Return the @a k-th parallel FE state space of the ParParametricBNLForm
       (const version). */
   const ParFiniteElementSpace *ParFESpace(int k) const;

   /// Return the @a k-th parallel FE parameters space of the ParParametricBNLForm.
   ParFiniteElementSpace *ParParamFESpace(int k);
   /** @brief Return the @a k-th parallel FE parameters space of the ParParametricBNLForm
       (const version). */
   const ParFiniteElementSpace *ParParamFESpace(int k) const;

   /** @brief Set he parallelel FE spaces for the state and the parametric fields.
    *  After a call to SetParSpaces(), the essential b.c. and the
       gradient-type (if different from the default) must be set again. */
   void SetParSpaces(Array<ParFiniteElementSpace *> &statef,
                     Array<ParFiniteElementSpace *> &paramf);

   /// Set the state essential BCs. Here, rhs is a true dof vector!
   virtual void SetEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                               Array<Vector *> &rhs);

   // Set the essential BCs for the parametric fields. Here, rhs is a true dof vector!
   virtual void SetParamEssentialBC(const Array<Array<int> *>&bdr_attr_is_ess,
                                    Array<Vector *> &rhs);


   /** @brief Calculates the residual for a state input given by block T-Vector.
    * The result is  Block T-Vector! The parametric fields should be set in advance
    * by calling SetParamFields(). */
   virtual void Mult(const Vector &x, Vector &y) const;

   /** @brief Calculates the product of the adjoint field and the derivative of
    * the state residual with respect to the parametric fields. The adjoint and
    * the state fields should be set in advance by calling SetAdjointFields() and
    * SetStateFields(). The input and the result are block T-Vectors!*/
   virtual void ParamMult(const Vector &x, Vector &y) const;

   /// Return the local block gradient matrix for the given true-dof vector x
   const BlockOperator &GetLocalGradient(const Vector &x) const;

   virtual BlockOperator &GetGradient(const Vector &x) const;

   /** @brief Set the operator type id for the blocks of the parallel gradient
       matrix/operator. The default type is Operator::Hypre_ParCSR. */
   void SetGradientType(Operator::Type tid);

   /// Destructor.
   virtual ~ParParametricBNLForm();

   /// Set the state fields
   virtual void SetStateFields(const Vector &xv) const;

   /// Set the adjoint fields
   virtual void SetAdjointFields(const Vector &av) const;

   /// Set the parameters/design fields
   virtual void SetParamFields(const Vector &dv) const;

};

}

#endif
#endif
