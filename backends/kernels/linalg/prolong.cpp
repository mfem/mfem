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
ProlongationOperator::ProlongationOperator(const kernels::kConformingProlongationOperator *P) :
   Operator(P->InLayout_(), P->OutLayout_()),
   pmat(P)
{
   dbg("\n\033[7m P->InLayout_().Size()=%d, P->OutLayout_()=%d\033[m",
          P->InLayout()->Size(), P->OutLayout()->Size());
   dbg("\n\033[7m Height()=%d, Width()=%d\033[m",Height(),Width());
   push();
   pop();
}


// **************************************************************************
void ProlongationOperator::Mult_(const kernels::Vector &x,
                                       kernels::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone()){
      const int N = y.Size();
      vector_op_set(N, x.GetData(), y.GetData());
      pop();
      return;
   }

   if (!kernels::config::Get().DoHostConformingProlongationOperator()){
      dbg("\n\033[35m[DEVICE::Mult]\033[m");
      pmat->d_Mult(x, y);
      pop();
      return;
   }else{
      dbg("\n\033[35m[HOST::Mult]\033[m");
   }
   assert(false);
   //x.Pull();
   //y.Pull(false);
   //pmat->Mult(x, y);
   //y.Push();
   pop();
}


// **************************************************************************
void ProlongationOperator::MultTranspose_(const kernels::Vector &x,
                                                kernels::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone()){
      const int N = y.Size();
      vector_op_set(N, x.GetData(), y.GetData());
      pop();
      return;
   }
   
   if (!kernels::config::Get().DoHostConformingProlongationOperator()){
      dbg("\n\033[35m[DEVICE::MultTranspose]\033[m");
      pmat->d_MultTranspose(x, y);
      pop();
      return;
   }else{
      dbg("\n\033[35m[HOST::MultTranspose]\033[m");
   }
   assert(false);
   //x.Pull();
   //y.Pull(false);
   //pmat->MultTranspose(x, y);
   //y.Push();
   pop();
}

// *****************************************************************************
/*void ProlongationOperator::Mult(const mfem::Vector &x,
                                      mfem::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone()){
      const int N = x.Size();
      const kernels::Vector &kx = x.Get_PVector()->As<const kernels::Vector>();
      kernels::Vector &ky = y.Get_PVector()->As<kernels::Vector>();
      vector_op_set(N, 
                    (const double*)kx.KernelsMem().ptr(),
                    (double*)ky.KernelsMem().ptr());
      pop();
      return;
   }else{
      assert(false);
   }
   
   if (pmat)   {
      // FIXME: create an OCCA version of 'pmat'
      x.Pull();
      y.Pull(false);
      pmat->Mult(x, y);
      y.Push();
   }
   else
   {
      multOp.Mult(x, y);
   }
   pop();
}

// *****************************************************************************
void ProlongationOperator::MultTranspose(const mfem::Vector &x,
                                               mfem::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone()){
      const int N = x.Size();
      const kernels::Vector &kx = x.Get_PVector()->As<const kernels::Vector>();
      const kernels::Vector &ky = y.Get_PVector()->As<const kernels::Vector>();
      vector_op_set(N,
                    (const double*)kx.KernelsMem().ptr(),
                    (double*)ky.KernelsMem().ptr());
      pop();
      return;
   }
   assert(false);
   if (pmat)
   {
      // FIXME: create an OCCA version of 'pmat'
      x.Pull();
      y.Pull(false);
      pmat->MultTranspose(x, y);
      y.Push();
   }
   else
   {
      multTransposeOp.Mult(x, y);
   }
   pop();
}
*/
} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
