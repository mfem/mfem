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

// *****************************************************************************
KernelsConstrainedOperator::KernelsConstrainedOperator(mfem::Operator *A_,
                                                       const mfem::Array<int> &constraintList_,
                                                       bool own_A_)

   : Operator(A_->InLayout()->As<Layout>()),
     z(OutLayout_()),
     w(OutLayout_()),
     mfem_z((z.DontDelete(), z)),
     mfem_w((w.DontDelete(), w))
{
   push();
   assert(A_->InLayout());
   Setup(OutLayout_().KernelsEngine().GetDevice(), A_, constraintList_, own_A_);
   pop();
}

// *****************************************************************************
void KernelsConstrainedOperator::Setup(kernels::device device_,
                                       mfem::Operator *A_,
                                       const mfem::Array<int> &constraintList_,
                                       bool own_A_)
{
   push();
   device = device_;
   A = A_;
   own_A = own_A_;
   constraintIndices = constraintList_.Size();
   if (constraintIndices)
   {
      assert(false);
      //constraintList.allocate(constraintIndices);
      assert(constraintList_.Get_PArray());
      //constraintList = constraintList_.Get_PArray()->As<Array>().KernelsMem();
   }
   //z.SetSize(height);
   //w.SetSize(height);
   dbg("done");
   pop();
}

// *****************************************************************************
void KernelsConstrainedOperator::EliminateRHS(const Vector &x, Vector &b) const
{
   push();
   w.Fill<double>(0.0);
   if (constraintIndices)
   {
      assert(false);
      vector_map_dofs(constraintIndices,
                      (double*)w.KernelsMem().ptr(),
                      (double*)x.KernelsMem().ptr(),
                      (int*)constraintList.ptr());
   }
   A->Mult(mfem_w, mfem_z);
   b.Axpby<double>(1.0, b, -1.0, z);
   if (constraintIndices)
   {
      assert(false);
      vector_map_dofs(constraintIndices,
                      (double*)b.KernelsMem().ptr(),
                      (double*)x.KernelsMem().ptr(),
                      (int*)constraintList.ptr());
   }
   pop();
}

// *****************************************************************************
void KernelsConstrainedOperator::Mult_(const Vector &x, Vector &y) const
{
   push();
   mfem::Vector mfem_y(y);
   if (constraintIndices == 0)
   {
      push();
      A->Mult(x.Wrap(), mfem_y);
      pop();
      return;
   }

   z.Assign<double>(x); // z = x

   vector_clear_dofs(constraintIndices, (double*)z.KernelsMem().ptr(),
                     (int*)constraintList.ptr());

   A->Mult(mfem_z, mfem_y);

   vector_map_dofs(constraintIndices,
                   (double*)y.KernelsMem().ptr(),
                   (double*)x.KernelsMem().ptr(),
                   (int*)constraintList.ptr());
   pop();
}

// *****************************************************************************
KernelsConstrainedOperator::~KernelsConstrainedOperator()
{
   if (own_A)
   {
      delete A;
   }
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
