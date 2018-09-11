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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem
{

namespace kernels
{


// **************************************************************************
ProlongationOperator::ProlongationOperator(const
                                           kernels::kConformingProlongationOperator *P) :
   Operator(P->InLayout_(), P->OutLayout_()),
   pmat(P)
{
   dbg("\033[7m P->InLayout_().Size()=%d, P->OutLayout_()=%d\033[m",
       P->InLayout()->Size(), P->OutLayout()->Size());
   dbg("\033[7m Height()=%d, Width()=%d\033[m",Height(),Width());
   push();
   pop();
}


// **************************************************************************
void ProlongationOperator::Mult_(const kernels::Vector &x,
                                 kernels::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone())
   {
      const int N = y.Size();
      vector_op_set(N, x.GetData(), y.GetData());
      pop();
      return;
   }

   if (!kernels::config::Get().DoHostConformingProlongationOperator())
   {
      dbg("\n\033[35m[DEVICE::Mult]\033[m");
      dbg("\n\033[35m[DEVICE::Mult] x.Size()=%d & y.Size()=%d\033[m",x.Size(),
          y.Size());
      pmat->d_Mult(x, y);
      pop();
      return;
   }
   else
   {
      dbg("\n\033[35m[HOST::Mult]\033[m");
      mfem::Vector my(y);
      pmat->Mult(x.Wrap(), my);
   }
   pop();
}


// **************************************************************************
void ProlongationOperator::MultTranspose_(const kernels::Vector &x,
                                          kernels::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone())
   {
      const int N = y.Size();
      vector_op_set(N, x.GetData(), y.GetData());
      pop();
      return;
   }

   if (!kernels::config::Get().DoHostConformingProlongationOperator())
   {
      dbg("\n\033[35m[DEVICE::MultTranspose]\033[m");
      dbg("\n\033[35m[DEVICE::MultTranspose] x.Size()=%d & y.Size()=%d\033[m",
          x.Size(),y.Size());
      pmat->d_MultTranspose(x, y);
      pop();
      return;
   }
   else
   {
      dbg("\n\033[35m[HOST::MultTranspose]\033[m");
      mfem::Vector my(y);
      pmat->MultTranspose(x.Wrap(), my);
   }
   pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
