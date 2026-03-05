// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MAGMA_LINALG
#define MFEM_MAGMA_LINALG

#include "batched.hpp"

#ifdef MFEM_USE_MAGMA

#include <magma_v2.h>

namespace mfem
{

class MagmaBatchedLinAlg : public BatchedLinAlgBase
{
public:
   void AddMult(const DenseTensor &A, const Vector &x, Vector &y,
                real_t alpha = 1.0, real_t beta = 1.0,
                Op op = Op::N) const override;
   void Invert(DenseTensor &A) const override;
   void LUFactor(DenseTensor &A, Array<int> &P) const override;
   void LUSolve(const DenseTensor &A, const Array<int> &P,
                Vector &x) const override;
};

/// Singleton class for interfacing with the MAGMA library.
class Magma
{
   magma_queue_t queue; ///< The default MAGMA queue.
   Magma(); ///< Initialize the MAGMA library.
   ~Magma(); ///< Finalize the MAGMA library.
   static Magma &Instance(); ///< Get the unique instance of this class.
public:
   /// Return the queue, creating it if needed.
   static magma_queue_t Queue();
};

} // namespace mfem

#endif

#endif
