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

namespace mfem
{

namespace raja
{

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

} // namespace mfem::raja

} // namespace mfem

#endif // MFEM_BACKENDS_RAJA_OPERATOR_HPP
