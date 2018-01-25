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

}

#endif // MFEM_USE_MPI

#endif
