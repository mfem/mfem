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
kConstrainedOperator::kConstrainedOperator(mfem::Operator *_A,
                                           const mfem::Array<int> &_constraintList,
                                           bool _own_A)
   : Operator(_A->InLayout()->As<kernels::Layout>()),
     engine(OutLayout_().KernelsEngine()),
     A(_A),
     own_A(_own_A),
     constraintList(_constraintList),
     constraintIndices(_constraintList.Size()),
     kz(OutLayout_()),
     kw(OutLayout_()),
     mfem_z((kz.DontDelete(), kz)),
     mfem_w((kw.DontDelete(), kw))
{
   push();
   if (constraintIndices>0)
   {
      constraintList.Resize(engine.MakeLayout(constraintIndices));
      constraintList=_constraintList;
      constraintList.Push();
      assert(constraintList.Get_PArray());
   }
   dbg("done");
   pop();
}


// *****************************************************************************
void kConstrainedOperator::EliminateRHS(const kernels::Vector &x,
                                              kernels::Vector &b) const
{
   push();
   
   kw.Fill<double>(0.0);
   
   if (constraintIndices)
   {
      const kernels::Array constraints_list =
         constraintList.Get_PArray()->As<const kernels::Array>();
      vector_map_dofs(constraintIndices,
                      (double*)kw.KernelsMem().ptr(),
                      (const double*)x.KernelsMem().ptr(),
                      (const int*) constraints_list.KernelsMem().ptr());
   }
   
   A->Mult(mfem_w, mfem_z);
   b.Axpby<double>(1.0, b, -1.0, kz);
   
   if (constraintIndices)
   {
      const kernels::Array constraints_list =
         constraintList.Get_PArray()->As<const kernels::Array>();
      vector_map_dofs(constraintIndices,
                      (double*)b.KernelsMem().ptr(),
                      (const double*)x.KernelsMem().ptr(),
                      (const int*) constraints_list.KernelsMem().ptr());
   }
   pop();
}

// *****************************************************************************
void kConstrainedOperator::Mult_(const kernels::Vector &x,
                                       kernels::Vector &y) const
{
   push();

   if (constraintIndices == 0)
   {
      mfem::Vector my(y);
      A->Mult(x.Wrap(), my);
      dbg("\033[7;1mdone");
      return;
   }
   
   // ex1pd comes here
   const kernels::Array &constraints_list =
      constraintList.Get_PArray()->As<const kernels::Array>();
   
   kz.Assign<double>(x); // z = x

   vector_clear_dofs(constraintIndices,
                     (double*)kz.KernelsMem().ptr(),
                     (const int*) constraints_list.KernelsMem().ptr());

   A->Mult(mfem_z, mfem_y);

   vector_map_dofs(constraintIndices,
                   (double*)y.KernelsMem().ptr(),
                   (const double*)x.KernelsMem().ptr(),
                   (const int*) constraints_list.KernelsMem().ptr());
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
