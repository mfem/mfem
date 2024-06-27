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

#ifndef MFEM_BATCHED_LINALG
#define MFEM_BATCHED_LINALG

#include "../../config/config.hpp"
#include "../densemat.hpp"
#include <array>
#include <memory>

namespace mfem
{

class BatchedLinAlg
{
public:
   enum Backend
   {
      NATIVE,
      GPU_BLAS,
      MAGMA,
      NUM_BACKENDS
   };
private:
   std::array<std::unique_ptr<class BatchedLinAlgBase>,
          Backend::NUM_BACKENDS> backends;
   Backend preferred_backend;
   BatchedLinAlg();
   static BatchedLinAlg &Instance();
public:
   static void Mult(const DenseTensor &A, const Vector &x, Vector &y);
   static void Invert(DenseTensor &A);
   static void LUFactor(DenseTensor &A, Array<int> &P);
   static void LUSolve(const DenseTensor &A, const Array<int> &P, Vector &x);

   static void SetPreferredBackend(Backend backend);
   static const BatchedLinAlgBase &Get(Backend backend);
};

class BatchedLinAlgBase
{
public:
   virtual void Mult(const DenseTensor &A, const Vector &x, Vector &y) const = 0;
   virtual void Invert(DenseTensor &A) const = 0;
   virtual void LUFactor(DenseTensor &A, Array<int> &P) const = 0;
   virtual void LUSolve(const DenseTensor &LU, const Array<int> &P,
                        Vector &x) const = 0;
   virtual ~BatchedLinAlgBase() { }
};

} // namespace mfem

#endif
