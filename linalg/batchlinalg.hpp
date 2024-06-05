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

class BatchSolver
{
public:
   enum struct SolveMode : int
   {
      LU,
      INVERSE
   };

   // Compute inverse of a batch of matrices in M; layout must be (height, width, num_mats)
   static void GetInverse(const DenseTensor &M, DenseTensor &M_inv);

private:
   BatchSolver::SolveMode mode_;

   bool setup_    = false;
   bool lu_valid_ = false;

   DenseTensor LUMatrixBatch_, InvMatrixBatch_;
   Array<int> P_;
   Array<double *> lu_ptr_array_;
   MemoryType d_mt_;

   int num_matrices_, matrix_size_;



   // for nvcc
public:
   //Compute LU or Inverse
   void Setup();

   void ComputeLU();

   void SolveLU(const Vector &b, Vector &x) const;

   void ComputeInverse(DenseTensor &InvMatBatch) const;
   void ApplyInverse(const Vector &b, Vector &x) const;
   
public:
   BatchSolver() = delete;

   BatchSolver(const SolveMode mode, MemoryType d_mt = MemoryType::DEFAULT);

   BatchSolver(const DenseTensor &MatrixBatch, const SolveMode mode,
               MemoryType d_mt = MemoryType::DEFAULT);

   void AssignMatrices(const DenseTensor &MatrixBatch);

   void AssignMatrices(const Vector &vMatrixBatch, const int size,
                       const int num_matrices);

   void GetInverse(DenseTensor &InvMatBatch) const;

   //Solve linear system Ax = b
   void Mult(const Vector &b, Vector &x) const;

   void ReleaseMemory();

};

} // namespace mfem

#endif
