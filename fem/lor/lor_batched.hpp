// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_BATCHED
#define MFEM_LOR_BATCHED

#include "lor.hpp"

namespace mfem
{

/// @brief Efficient batched assembly of LOR discretizations on device.
///
/// This class should typically be used by the user-facing classes
/// LORDiscretization and ParLORDiscretization. Only certain bilinear forms are
/// supported, currently:
///
///  - H1 diffusion + mass
///  - ND curl-curl + mass (2D only)
class BatchedLORAssembly
{
protected:
   FiniteElementSpace &fes_ho; ///< The high-order space.
   const Array<int> &ess_dofs; ///< Essential DOFs to eliminate.

   Vector X_vert; ///< LOR vertex coordinates.

   /// Get the vertices of the LOR mesh and place the result in @a X_vert.
   void GetLORVertexCoordinates();

   /// @brief The elementwise LOR matrices in a sparse "ij" format.
   ///
   /// This is interpreted to have shape (nnz_per_row, ndof_per_el, nel_ho). For
   /// index (i, j, k), this represents row @a j of the @a kth element matrix.
   /// The column index is given by sparse_mapping(i, j).
   Vector sparse_ij;

   /// @brief The sparsity pattern of the element matrices.
   ///
   /// For local DOF index @a j, sparse_mapping(i, j) is the column index of the
   /// @a ith nonzero in the @a jth row. If the index is negative, that entry
   /// should be skipped (there is no corresponding nonzero).
   DenseMatrix sparse_mapping;

public:
   /// Does the given form support batched assembly?
   static bool FormIsSupported(BilinearForm &a);

   /// @brief Assemble the given form as a matrix and place the result in @a A.
   ///
   /// In serial, the result will be a SparseMatrix. In parallel, the result
   /// will be a HypreParMatrix.
   static void Assemble(BilinearForm &a,
                        FiniteElementSpace &fes_ho,
                        const Array<int> &ess_dofs,
                        OperatorHandle &A);

protected:
   /// After assembling the "sparse IJ" format, convert it to CSR.
   void SparseIJToCSR(OperatorHandle &A) const;

   /// Assemble the system without eliminating essential DOFs.
   void AssembleWithoutBC(OperatorHandle &A);

   /// Called by one of the specialized classes, e.g. BatchedLORDiffusion.
   BatchedLORAssembly(BilinearForm &a,
                      FiniteElementSpace &fes_ho_,
                      const Array<int> &ess_dofs_);

   virtual ~BatchedLORAssembly() { }

   /// Return the first domain integrator in the form @a i of type @a T.
   template <typename T>
   static T *GetIntegrator(BilinearForm &a)
   {
      Array<BilinearFormIntegrator*> *integs = a.GetDBFI();
      if (integs != NULL)
      {
         for (auto *i : *integs)
         {
            if (auto *ti = dynamic_cast<T*>(i))
            {
               return ti;
            }
         }
      }
      return nullptr;
   }

   // Compiler limitation: these should be protected, but they contain
   // MFEM_FORALL kernels, and so they must be public.
public:
   /// @brief Fill the I array of the sparse matrix @a A.
   ///
   /// @note AssemblyKernel must be called first to populate @a sparse_mapping.
   int FillI(SparseMatrix &A) const;

   /// @brief Fill the J and data arrays of the sparse matrix @a A.
   ///
   /// @note AssemblyKernel must be called first to populate @a sparse_mapping
   /// and @a sparse_ij.
   void FillJAndData(SparseMatrix &A) const;

#ifdef MFEM_USE_MPI
   /// Assemble the system in parallel and place the result in @a A.
   void ParAssemble(OperatorHandle &A);
#endif

   /// Assemble the system, and place the result in @a A.
   void Assemble(OperatorHandle &A);

   /// @brief Pure virtual function for the kernel actually performing the
   /// assembly. Overridden in the derived classes.
   virtual void AssemblyKernel() = 0;
};

template <typename T>
void EnsureCapacity(Memory<T> &mem, int capacity)
{
   if (mem.Capacity() < capacity)
   {
      mem.Delete();
      mem.New(capacity, mem.GetMemoryType());
   }
}

#ifdef MFEM_USE_MPI

/// @brief Make @a A_hyp steal ownership of its diagonal part @a A_diag.
///
/// If @a A_hyp does not own I and J, then they are aliases pointing to the I
/// and J arrays in @a A_diag. In that case, this function swaps the memory
/// objects. Similarly for the data array.
///
/// After this function is called, @a A_hyp will own all of the arrays of its
/// diagonal part.
///
/// @note I and J can only be aliases when HYPRE_BIGINT is disabled.
void HypreStealOwnership(HypreParMatrix &A_hyp, SparseMatrix &A_diag);

#endif

}

#endif
