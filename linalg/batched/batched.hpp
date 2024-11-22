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
///
/// The static member functions will delegate to the active backend (which can
/// be set using SetActiveBackend(), see BatchedLinAlg::Backend for all
/// available backends and the order in which they will be chosen initially).
/// Operations can be performed directly with a specific backend using Get().
class BatchedLinAlg
{
public:
   /// @brief Available backends for implementations of batched algorithms.
   ///
   /// The initially active backend will be the first available backend in this
   /// order: MAGMA, GPU_BLAS, NATIVE.
   enum Backend
   {
      /// @brief The standard MFEM backend, implemented using mfem::forall
      /// kernels. Not as performant as the other kernels.
      NATIVE,
      /// @brief Either cuBLAS or hipBLAS, depending on whether MFEM is using
      /// CUDA or HIP. Not available otherwise.
      GPU_BLAS,
      /// MAGMA backend, only available if MFEM is compiled with MAGMA support.
      MAGMA,
      /// Counter for the number of backends.
      NUM_BACKENDS
   };

   /// Operation type (transposed or not transposed)
   enum Op
   {
      N, ///< Not transposed.
      T  ///< Transposed.
   };

private:
   /// All available backends. Unavailble backends will be nullptr.
   std::array<std::unique_ptr<class BatchedLinAlgBase>,
          Backend::NUM_BACKENDS> backends;
   Backend active_backend;
   /// Default constructor. Private.
   BatchedLinAlg();
   /// Return the singleton instance.
   static BatchedLinAlg &Instance();
public:
   /// @brief Computes $y = \alpha A^{op} x + \beta y$.
   ///
   /// $A^{op}$ is either $A$ or $A^T$ depending on the value of @a op.
   /// $A$ is a block diagonal matrix, represented by the DenseTensor @a A with
   /// shape (m, n, n_mat). $x$ has shape (tr?m:n, k, n_mat), and $y$ has shape
   /// (tr?n:m, k, n_mat), where 'tr' is true in the transposed case.
   static void AddMult(const DenseTensor &A, const Vector &x, Vector &y,
                       real_t alpha = 1.0, real_t beta = 1.0,
                       Op op = Op::N);
   /// Computes $y = A x$ (e.g. by calling @ref AddMult "AddMult(A,x,y,1,0,Op::N)").
   static void Mult(const DenseTensor &A, const Vector &x, Vector &y);
   /// Computes $y = A^T x$ (e.g. by calling @ref AddMult "AddMult(A,x,y,1,0,Op::T)").
   static void MultTranspose(const DenseTensor &A, const Vector &x, Vector &y);
   /// @brief Replaces the block diagonal matrix $A$ with its inverse $A^{-1}$.
   ///
   /// $A$ is represented by the DenseTensor @a A with shape (m, m, n_mat).
   static void Invert(DenseTensor &A);
   /// @brief Replaces the block diagonal matrix $A$ with its LU factors. The
   /// pivots are stored in @a P.
   ///
   /// $A$ is represented by the DenseTensor @a A with shape (n, n, n_mat). On
   /// output, $P$ has shape (n, n_mat).
   static void LUFactor(DenseTensor &A, Array<int> &P);
   /// @brief Replaces $x$ with $A^{-1} x$, given the LU factors @a A and pivots
   /// @a P of the block-diagonal matrix $A$.
   ///
   /// The LU factors and pivots of $A$ should be obtained by first calling
   /// LUFactor(). $A$ has shape (n, n, n_mat) and $x$ has shape (n, n_rhs,
   /// n_mat).
   ///
   /// @warning LUSolve() and LUFactor() should be called using the same backend
   /// because of potential incompatibilities (e.g. 0-based or 1-based
   /// indexing).
   static void LUSolve(const DenseTensor &A, const Array<int> &P, Vector &x);
   /// @brief Returns true if the requested backend is available.
   ///
   /// The available backends depend on which third-party libraries MFEM is
   /// compiled with, and whether the the CUDA/HIP device is enabled.
   static bool IsAvailable(Backend backend);
   /// Set the default backend for batched linear algebra operations.
   static void SetActiveBackend(Backend backend);
   /// Get the default backend for batched linear algebra operations.
   static Backend GetActiveBackend();
   /// @brief Get the BatchedLinAlgBase object associated with a specific
   /// backend.
   ///
   /// This allows the user to perform specific operations with a backend
   /// different from the active backend.
   static const BatchedLinAlgBase &Get(Backend backend);
};

/// Abstract base clase for batched linear algebra operations.
class BatchedLinAlgBase
{
public:
   using Op = BatchedLinAlg::Op;
   /// See BatchedLinAlg::AddMult.
   virtual void AddMult(const DenseTensor &A, const Vector &x, Vector &y,
                        real_t alpha = 1.0, real_t beta = 1.0,
                        Op op = Op::N) const = 0;
   /// See BatchedLinAlg::Mult.
   virtual void Mult(const DenseTensor &A, const Vector &x, Vector &y) const;
   /// See BatchedLinAlg::MultTranspose.
   virtual void MultTranspose(const DenseTensor &A, const Vector &x,
                              Vector &y) const;
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
