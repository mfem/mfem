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

#ifndef MFEM_NATIVE_LINALG
#define MFEM_NATIVE_LINALG

#include "batched.hpp"

namespace mfem
{

class NativeBatchedLinAlg : public BatchedLinAlgBase
{
public:
   void AddMult(const DenseTensor &A, const Vector &x, Vector &y,
                real_t alpha, real_t beta, Op op) const override;
   void Invert(DenseTensor &A) const override;
   void LUFactor(DenseTensor &A, Array<int> &P) const override;
   void LUSolve(const DenseTensor &LU, const Array<int> &P,
                Vector &x) const override;
};

} // namespace mfem

#endif
