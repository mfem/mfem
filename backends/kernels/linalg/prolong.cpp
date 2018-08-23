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
void KernelsMult(const mfem::Operator &op, const kernels::Vector &x,
                 kernels::Vector &y)
{
   push();assert(false);
   kernels::device device = x.KernelsLayout().KernelsEngine().GetDevice();
   if (device.hasSeparateMemorySpace()) {
      assert(false);
      /*mfem::Vector &hostX = GetHostVector(0, op.Width());
      mfem::Vector &hostY = GetHostVector(1, op.Height());
      x.KernelsMem().copyTo(hostX.GetData(), hostX.Size() * sizeof(double));
      op.Mult(hostX, hostY);
      y.KernelsMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));*/
   }
   else
   {
      
      //dbg("[KernelsMult] op:%dx%d, x:%d, y:%d", op.Width(), op.Height(), x.Size(), y.Size());
      mfem::Vector hostX((double*) x.KernelsMem().ptr(), op.Width());
      //dbg("[KernelsMult] hostX:\n");hostX.Print();      
      //hostX.Pull();      
      x.KernelsMem().copyTo(hostX.GetData(), hostX.Size() * sizeof(double));
      //x.KernelsMem().copyFrom(hostX.GetData(), hostX.Size() * sizeof(double));
      
      mfem::Vector hostY((double*) y.KernelsMem().ptr(), op.Height());
      op.Mult(hostX, hostY);
      
      //hostY.Pull(false);
      
      //dbg("[KernelsMult] hostY:\n");hostY.Print();
      //assert(false);
      //y.KernelsMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));
      //y.KernelsMem().copyTo(hostY.GetData(), hostY.Size() * sizeof(double));
      hostY.Push();
      //y.Push();
      //dbg("[KernelsMult] y:\n");y.Print();
     }
   pop();
}

// *****************************************************************************
void KernelsMultTranspose(const mfem::Operator &op,
                          const kernels::Vector &x, kernels::Vector &y)
{
   push();assert(false);
   kernels::device device = x.KernelsLayout().KernelsEngine().GetDevice();
   if (device.hasSeparateMemorySpace())
   {
      assert(false);
      /*
        mfem::Vector &hostX = GetHostVector(1, op.Height());
        mfem::Vector &hostY = GetHostVector(0, op.Width());
        x.KernelsMem().copyTo((void*)hostX.GetData(), hostX.Size() * sizeof(double));
        op.MultTranspose(hostX, hostY);
        y.KernelsMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));*/
   }
   else
   {
      //dbg("[KernelsMultTranspose] op:%dx%d, x:%d, y:%d", op.Height(), op.Width(), x.Size(), y.Size());
      mfem::Vector hostX((double*) x.KernelsMem().ptr(), op.Height());
      //hostX.Pull();
      x.KernelsMem().copyTo(hostX.GetData(), hostX.Size() * sizeof(double));
      //x.KernelsMem().copyFrom(hostX.GetData(), hostX.Size() * sizeof(double));
      mfem::Vector hostY((double*) y.KernelsMem().ptr(), op.Width());
      //hostY.Pull(false);
      op.MultTranspose(hostX, hostY);
      // -0.1307 -5.55112e-17 0.1307 -0.14772 -4.85723e-17 0.14772 -0.1307 -6.93889e-17
      // 0.1307 0.0378933 -5.55112e-17 0.0757866 -0.40912 -0.0378933 0.40912 -0.0757866
      // -5.55112e-17 0.0378933 -0.40912 0.40912 -0.0378933 0.151573 -0.151573 0.151573
      // -0.151573
      //assert(false);
      //y.KernelsMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));
      //y.KernelsMem().copyTo(hostY.GetData(), hostY.Size() * sizeof(double));
      hostY.Push();
      //dbg("[KernelsMultTranspose] y:\n");y.Print();
   }
   pop();
}

// **************************************************************************
ProlongationOperator::ProlongationOperator(KernelsSparseMatrix &multOp_,
                                           KernelsSparseMatrix &multTransposeOp_) :
   Operator(multOp_),
   pmat(NULL),
   multOp(multOp_),
   multTransposeOp(multTransposeOp_) {}

// **************************************************************************
ProlongationOperator::ProlongationOperator(Layout &in_layout,
                                           Layout &out_layout,
                                           const mfem::Operator *pmat_) :
   Operator(in_layout, out_layout),
   pmat(pmat_),
   multOp(*this),
   multTransposeOp(*this) {}

// **************************************************************************
void ProlongationOperator::Mult_(const kernels::Vector &x,
                                       kernels::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone()){
      const int N = y.Size();
      vector_op_set(N, 
                    (const double*)x.KernelsMem().ptr(),
                    (double*)y.KernelsMem().ptr());
   }
   pop();
}

// **************************************************************************
void ProlongationOperator::MultTranspose_(const kernels::Vector &x,
                                                kernels::Vector &y) const
{
   push();
   if (kernels::config::Get().IAmAlone()){
      const int N = y.Size();
      vector_op_set(N,
                    (const double*)x.KernelsMem().ptr(),
                    (double*)y.KernelsMem().ptr());

   }
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
