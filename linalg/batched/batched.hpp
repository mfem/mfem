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

/// @brief Class for performing batched linear algebra operations, potentially
/// using accelerated algorithms (GPU BLAS or MAGMA). Accessed using static
/// member functions.
class BatchedLinAlg
{
public:
   /// @brief Available backends for implementations of batched algorithms.
   ///
   /// The preferred backend will be the first available backend in this order:
   /// MAGMA, GPU_BLAS, NATIVE.
   enum Backend
   {
      /// @brief The standard MFEM backend, implemented using mfem::forall
      /// kernels. Not as performant as the other kernels.
      NATIVE,
      /// @brief Either cuBLAS or hipBLAS, depending on whether MFEM is using
      /// CUDA or HIP. Not available otherwise.
      GPU_BLAS,
      /// Magma backend, only available if MFEM is compiled with Magma support.
      MAGMA,
      /// Counter for the number of backends.
      NUM_BACKENDS
   };
private:
   /// All available backends. Unavailble backends will be nullptr.
   std::array<std::unique_ptr<class BatchedLinAlgBase>,
          Backend::NUM_BACKENDS> backends;
   Backend preferred_backend;
   /// Default constructor. Private.
   BatchedLinAlg();
   /// Return the singleton instance.
   static BatchedLinAlg &Instance();
public:
   static void AddMult(const DenseTensor &A, const Vector &x, Vector &y,
                       real_t alpha = 1.0, real_t beta = 1.0);
   static void Mult(const DenseTensor &A, const Vector &x, Vector &y);
   static void Invert(DenseTensor &A);
   static void LUFactor(DenseTensor &A, Array<int> &P);
   static void LUSolve(const DenseTensor &A, const Array<int> &P, Vector &x);
   static bool IsAvailable(Backend backend);
   /// Set the default backend for batched linear algebra operations.
   static void SetPreferredBackend(Backend backend);
   /// Get the default backend for batched linear algebra operations.
   static Backend GetPreferredBackend();
   /// @brief Get the BatchedLinAlgBase object associated with a specific
   /// backend.
   ///
   /// This allows the user to perform specific operations with a backend
   /// different from the preferred backend.
   static const BatchedLinAlgBase &Get(Backend backend);
};

/// Abstract base clase for batched linear algebra operations.
class BatchedLinAlgBase
{
public:
   /// See BatchedLinAlg::AddMult.
   virtual void AddMult(const DenseTensor &A, const Vector &x, Vector &y,
                        real_t alpha = 1.0, real_t beta = 1.0) const = 0;
   /// See BatchedLinAlg::Mult.
   virtual void Mult(const DenseTensor &A, const Vector &x, Vector &y) const;
   /// See BatchedLinAlg::Invert.
   virtual void Invert(DenseTensor &A) const = 0;
   /// See BatchedLinAlg::LUFactor.
   virtual void LUFactor(DenseTensor &A, Array<int> &P) const = 0;
   /// See BatchedLinAlg::LUSolve.
   virtual void LUSolve(const DenseTensor &LU, const Array<int> &P,
                        Vector &x) const = 0;
   /// Virtual destructor.
   virtual ~BatchedLinAlgBase() { }
};

} // namespace mfem

#endif
