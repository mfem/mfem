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

#ifndef MFEM_BACKENDS_KERNELS_CONSTRAINED_OPERATOR_HPP
#define MFEM_BACKENDS_KERNELS_CONSTRAINED_OPERATOR_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

// **************************************************************************
class kConstrainedOperator : public Operator
{
protected:
   const kernels::Engine &engine;
   mfem::Operator *A;              //< The unconstrained Operator.
   bool own_A;                     //< Ownership flag for A.
   mfem::Array<int> constraintList; //< List of constrained indices/dofs.
   int constraintIndices;
   mutable kernels::Vector kz, kw;      //< Auxiliary vectors.
   mutable mfem::Vector mfem_y, mfem_z, mfem_w; // Wrap z, w
public:
   kConstrainedOperator(mfem::Operator*, const mfem::Array<int>&, bool = false);


   /** @brief Eliminate "essential boundary condition" values specified in @a x
       from the given right-hand side @a b.

       Performs the following steps:

       z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   void EliminateRHS(const kernels::Vector &x, kernels::Vector &b) const;

   /** @brief Constrained operator action.

       Performs the following steps:

       z = A((x_i,0));  y_i = z_i;  y_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   virtual void Mult_(const kernels::Vector &x, kernels::Vector &y) const;

   // Destructor: destroys the unconstrained Operator @a A if @a own_A is true.
   virtual ~kConstrainedOperator();
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_CONSTRAINED_OPERATOR_HPP
