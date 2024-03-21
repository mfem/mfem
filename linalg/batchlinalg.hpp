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

#ifndef MFEM_BATCHLINALG
#define MFEM_BATCHLINALG

#include "../config/config.hpp"
#include "../general/globals.hpp"
#include "matrix.hpp"
#include "densemat.hpp"

namespace mfem
{

class LibBatchSolver
{
public:
   enum struct SolveMode : int
   {
      LU,
      INVERSE
   };

   // Compute inverse of a batch of matrices in M; layout must be (height, width, num_mats)
   static void GetInverse(const mfem::DenseTensor &M, mfem::DenseTensor &M_inv);

private:
   LibBatchSolver::SolveMode mode_;

   bool setup_    = false;
   bool lu_valid_ = false;

   mfem::DenseTensor LUMatrixBatch_, InvMatrixBatch_;
   mfem::Array<int> P_;
   mfem::Array<double *> lu_ptr_array_;

   int num_matrices_, matrix_size_;

   void ApplyInverse(const mfem::Vector &b, mfem::Vector &x) const;

   // for nvcc
public:
   //Compute LU or Inverse
   void Setup();

   void ComputeLU();

   void SolveLU(const mfem::Vector &b, mfem::Vector &x) const;

   void ComputeInverse(mfem::DenseTensor &InvMatBatch) const;

public:
   LibBatchSolver() = delete;

   LibBatchSolver(const SolveMode mode);

   LibBatchSolver(const mfem::DenseTensor &MatrixBatch, const SolveMode mode);

   void AssignMatrices(const mfem::DenseTensor &MatrixBatch);

   void AssignMatrices(const mfem::Vector &vMatrixBatch, const int ndofs,
                       const int num_matrices);

   void GetInverse(mfem::DenseTensor &InvMatBatch) const;

   //Solve linear system Ax = b
   void Mult(const mfem::Vector &b, mfem::Vector &x) const;

   void ReleaseMemory();

};  // namespace lin_alg

} // namespace mfem

#endif
