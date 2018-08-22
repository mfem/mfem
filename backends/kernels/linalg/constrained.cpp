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
kConstrainedOperator::kConstrainedOperator(mfem::Operator *A_,
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
void kConstrainedOperator::Setup(kernels::device device_,
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
     assert(constraintList_.Get_PArray());
     constraintList = constraintList_.Get_PArray()->As<Array>().KernelsMem();
   }
   pop();
}

// *****************************************************************************
void kConstrainedOperator::EliminateRHS(const kernels::Vector &x,
                                              kernels::Vector &b) const
{
   push();//assert(false); // ex1pd comes here, Laghos does not
   
   w.Fill<double>(0.0);
   if (constraintIndices)
   {
      //assert(false); // ex1pd comes here
      vector_map_dofs(constraintIndices,
                      (double*)w.KernelsMem().ptr(),
                      (double*)x.KernelsMem().ptr(),
                      (int*)constraintList.ptr());
   }
   A->Mult(mfem_w, mfem_z);
   b.Axpby<double>(1.0, b, -1.0, z);
   if (constraintIndices)
   {
      //assert(false); // ex1pd comes here
      vector_map_dofs(constraintIndices,
                      (double*)b.KernelsMem().ptr(),
                      (double*)x.KernelsMem().ptr(),
                      (int*)constraintList.ptr());
   }

   pop();
}

// *****************************************************************************
void kConstrainedOperator::Mult_(const kernels::Vector &x,
                                       kernels::Vector &y) const
{
   push();
   dbg("x.Size()=%d",x.Size());
   dbg("y.Size()=%d",y.Size());
   dbg("A->InLayout().Size()=%d",A->InLayout()->Size());
   dbg("A->OutLayout().Size()=%d",A->OutLayout()->Size());
   
   if (constraintIndices == 0)
   {
      mfem::Vector my(y);
      // linalg/operator.hpp:441
      // P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y);
      // kprolong.cpp:177
      A->Mult(x.Wrap(), my);
      dbg("\033[7;1mdone");
      return;
   }
   //assert(false); // ex1pd comes here
   z.Assign<double>(x); // z = x

   vector_clear_dofs(constraintIndices,
                     (double*)z.KernelsMem().ptr(),
                     (int*)constraintList.ptr());

   A->Mult(mfem_z, mfem_y);

   vector_map_dofs(constraintIndices,
                   (double*)y.KernelsMem().ptr(),
                   (double*)x.KernelsMem().ptr(),
                   (int*)constraintList.ptr());
}

// *****************************************************************************
kConstrainedOperator::~kConstrainedOperator()
{
   if (own_A)
   {
      delete A;
   }
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
