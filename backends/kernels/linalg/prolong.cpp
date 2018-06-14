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

mfem::Vector& GetHostVector(const int id, const int64_t size)
{
   push();
   static std::vector<mfem::Vector*> v;
   if (v.size() <= (size_t) id)
   {
      for (int i = (int) v.size(); i < (id + 1); ++i)
      {
         v.push_back(new mfem::Vector);
      }
   }
   if (size >= 0)
   {
      v[id]->SetSize(size);
   }
   pop();
   return *(v[id]);
}

void KernelsMult(const mfem::Operator &op,
              const Vector &x, Vector &y)
{
   push();
   kernels::device device = x.KernelsLayout().KernelsEngine().GetDevice();
   if (device.hasSeparateMemorySpace())
   {
      assert(false);/*
                      mfem::Vector &hostX = GetHostVector(0, op.Width());
                      mfem::Vector &hostY = GetHostVector(1, op.Height());
                      x.KernelsMem().copyTo(hostX.GetData(), hostX.Size() * sizeof(double));
                      op.Mult(hostX, hostY);
                      y.KernelsMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));*/
   }
   else
   {
      mfem::Vector hostX((double*) x.KernelsMem().ptr(), x.Size());
      mfem::Vector hostY((double*) y.KernelsMem().ptr(), y.Size());
      op.Mult(hostX, hostY);
   }
   pop();
}

void KernelsMultTranspose(const mfem::Operator &op,
                       const Vector &x, Vector &y)
{
   push();
   kernels::device device = x.KernelsLayout().KernelsEngine().GetDevice();
   if (device.hasSeparateMemorySpace())
   {
      assert(false);/*
      mfem::Vector &hostX = GetHostVector(1, op.Height());
      mfem::Vector &hostY = GetHostVector(0, op.Width());
      x.KernelsMem().copyTo((void*)hostX.GetData(), hostX.Size() * sizeof(double));
      op.MultTranspose(hostX, hostY);
      y.KernelsMem().copyFrom(hostY.GetData(), hostY.Size() * sizeof(double));*/
   }
   else
   {
      mfem::Vector hostX((double*) x.KernelsMem().ptr(), x.Size());
      mfem::Vector hostY((double*) y.KernelsMem().ptr(), y.Size());
      op.MultTranspose(hostX, hostY);
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
void ProlongationOperator::Mult_(const Vector &x, Vector &y) const
{
   push();
   //assert(false);
   if (pmat)
   {
      dbg("\033[31;7mpmat %dx%d",pmat->Width(),pmat->Height());
      KernelsMult(*pmat, x, y);
   }
   else
   {
      // TODO: define 'ox' and 'oy'
      multOp.Mult_(x, y);
   }
   pop();
}

// **************************************************************************
void ProlongationOperator::MultTranspose_(const Vector &x, Vector &y) const
{
   push();
   if (pmat)
   {
      KernelsMultTranspose(*pmat, x, y);
   }
   else
   {
      multTransposeOp.Mult_(x, y);
   }
   pop();
}

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
