// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "batched.hpp"
#include "native.hpp"
#include "gpu_blas.hpp"
#include "magma.hpp"

namespace mfem
{

BatchedLinAlg::BatchedLinAlg()
{
   backends[NATIVE].reset(new NativeBatchedLinAlg);

   if (Device::Allows(mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK))
   {
#ifdef MFEM_USE_CUDA_OR_HIP
      backends[GPU_BLAS].reset(new GPUBlasBatchedLinAlg);
#endif

#ifdef MFEM_USE_MAGMA
      backends[MAGMA].reset(new MagmaBatchedLinAlg);
#endif

#if defined(MFEM_USE_MAGMA)
      active_backend = MAGMA;
#elif defined(MFEM_USE_CUDA_OR_HIP)
      active_backend = GPU_BLAS;
#else
      active_backend = NATIVE;
#endif
   }
   else
   {
      active_backend = NATIVE;
   }
}

BatchedLinAlg &BatchedLinAlg::Instance()
{
   static BatchedLinAlg instance;
   return instance;
}

void BatchedLinAlg::AddMult(const DenseTensor &A, const Vector &x, Vector &y,
                            real_t alpha, real_t beta, Op op)
{
   Get(Instance().active_backend).AddMult(A, x, y, alpha, beta, op);
}

void BatchedLinAlg::Mult(const DenseTensor &A, const Vector &x, Vector &y)
{
   Get(Instance().active_backend).Mult(A, x, y);
}

void BatchedLinAlg::MultTranspose(const DenseTensor &A, const Vector &x,
                                  Vector &y)
{
   Get(Instance().active_backend).MultTranspose(A, x, y);
}

void BatchedLinAlg::Invert(DenseTensor &A)
{
   Get(Instance().active_backend).Invert(A);
}

void BatchedLinAlg::LUFactor(DenseTensor &A, Array<int> &P)
{
   Get(Instance().active_backend).LUFactor(A, P);
}

void BatchedLinAlg::LUSolve(const DenseTensor &A, const Array<int> &P,
                            Vector &x)
{
   Get(Instance().active_backend).LUSolve(A, P, x);
}

bool BatchedLinAlg::IsAvailable(BatchedLinAlg::Backend backend)
{
   return Instance().backends[backend] != nullptr;
}

void BatchedLinAlg::SetActiveBackend(BatchedLinAlg::Backend backend)
{
   MFEM_VERIFY(IsAvailable(backend), "Requested backend not supported.");
   Instance().active_backend = backend;
}

BatchedLinAlg::Backend BatchedLinAlg::GetActiveBackend()
{
   return Instance().active_backend;
}

const BatchedLinAlgBase &BatchedLinAlg::Get(BatchedLinAlg::Backend backend)
{
   auto &backend_ptr = Instance().backends[backend];
   MFEM_VERIFY(backend_ptr, "Requested backend not supported.")
   return *backend_ptr;
}

void BatchedLinAlgBase::Mult(const DenseTensor &A, const Vector &x,
                             Vector &y) const
{
   AddMult(A, x, y, 1.0, 0.0);
}

void BatchedLinAlgBase::MultTranspose(const DenseTensor &A, const Vector &x,
                                      Vector &y) const
{
   AddMult(A, x, y, 1.0, 0.0, Op::T);
}

}
