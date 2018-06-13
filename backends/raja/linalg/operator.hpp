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

#ifndef MFEM_BACKENDS_RAJA_OPERATOR_HPP
#define MFEM_BACKENDS_RAJA_OPERATOR_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

// **************************************************************************
class Operator : public mfem::Operator
{
public:
   /// Creare an operator with the same dimensions as @a orig.
   Operator(const Operator &orig)
      : mfem::Operator(orig) { }

   Operator(Layout &layout)
      : mfem::Operator(layout) { }

   Operator(Layout &in_layout, Layout &out_layout)
      : mfem::Operator(in_layout, out_layout) { }

   Layout &InLayout_() const
   { return *static_cast<Layout*>(in_layout.Get()); }

   Layout &OutLayout_() const
   { return *static_cast<Layout*>(out_layout.Get()); }

   virtual void Mult_(const Vector &x, Vector &y) const = 0;

   virtual void MultTranspose_(const Vector &x, Vector &y) const
   { MFEM_ABORT("method is not supported"); }

   // override
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const
   {
      Mult_(x.Get_PVector()->As<Vector>(),
            y.Get_PVector()->As<Vector>());
   }

   // override
   virtual void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
   {
      MultTranspose_(x.Get_PVector()->As<Vector>(),
                     y.Get_PVector()->As<Vector>());
   }
};


// **************************************************************************
class RajaConstrainedOperator : public Operator
{
protected:
   raja::device device;

   mfem::Operator *A;              //< The unconstrained Operator.
   bool own_A;                     //< Ownership flag for A.
   raja::memory constraintList;    //< List of constrained indices/dofs.
   int constraintIndices;
   mutable raja::Vector z, w;      //< Auxiliary vectors.
   mutable mfem::Vector mfem_z, mfem_w; // Wrap z, w

public:
   /** @brief Constructor from a general Operator and a list of essential
       indices/dofs.

       Specify the unconstrained operator @a *A and a @a list of indices to
       constrain, i.e. each entry @a list[i] represents an essential-dof. If the
       ownership flag @a own_A is true, the operator @a *A will be destroyed
       when this object is destroyed. */
   RajaConstrainedOperator(mfem::Operator *A_,
                           const mfem::Array<int> &constraintList_,
                           bool own_A_ = false);

   void Setup(raja::device device_,
              mfem::Operator *A_,
              const mfem::Array<int> &constraintList_,
              bool own_A_ = false);

   /** @brief Eliminate "essential boundary condition" values specified in @a x
       from the given right-hand side @a b.

       Performs the following steps:

       z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   void EliminateRHS(const Vector &x, Vector &b) const;

   /** @brief Constrained operator action.

       Performs the following steps:

       z = A((x_i,0));  y_i = z_i;  y_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   virtual void Mult_(const Vector &x, Vector &y) const;

   // Destructor: destroys the unconstrained Operator @a A if @a own_A is true.
   virtual ~RajaConstrainedOperator();
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_OPERATOR_HPP
