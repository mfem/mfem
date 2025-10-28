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

#ifndef MFEM_LOR_BATCHED
#define MFEM_LOR_BATCHED

#include "lor.hpp"
#include "../qspace.hpp"

namespace mfem
{

/// @brief Efficient batched assembly of LOR discretizations on device.
///
/// This class should typically be used by the user-facing classes
/// LORDiscretization and ParLORDiscretization. Only certain bilinear forms are
/// supported, currently:
///
///  - H1 diffusion + mass
///  - DG diffusion + mass (in progress)
///  - ND curl-curl + mass
///  - RT div-div + mass
///
/// Whether a form is supported can be checked with the static member function
/// BatchedLORAssembly::FormIsSupported.
class BatchedLORAssembly
{
protected:
   FiniteElementSpace &fes_ho; ///< The high-order space.

   Vector X_vert; ///< LOR vertex coordinates.

   /// @brief The elementwise LOR matrices in a sparse "ij" format.
   ///
   /// This is interpreted to have shape (nnz_per_row, ndof_per_el, nel_ho). For
   /// index (i, j, k), this represents row @a j of the @a kth element matrix.
   /// The column index is given by sparse_mapping(i, j).
   Vector sparse_ij;

   /// @brief The sparsity pattern of the element matrices.
   ///
   /// This array should be interpreted as having shape (nnz_per_row,
   /// ndof_per_el). For local DOF index @a j, sparse_mapping(i, j) is the
   /// column index of the @a ith nonzero in the @a jth row. If the index is
   /// negative, that entry should be skipped (there is no corresponding
   /// nonzero).
   Array<int> sparse_mapping;

public:
   /// Construct the batched assembly object corresponding to @a fes_ho_.
   BatchedLORAssembly(FiniteElementSpace &fes_ho_);

   /// Returns true if the form @a a supports batched assembly, false otherwise.
   static bool FormIsSupported(BilinearForm &a);

   /// @brief Assemble the given form as a matrix and place the result in @a A.
   ///
   /// In serial, the result will be a SparseMatrix. In parallel, the result
   /// will be a HypreParMatrix.
   void Assemble(BilinearForm &a, const Array<int> ess_dofs, OperatorHandle &A);

   /// Compute the vertices of the LOR mesh and place the result in @a X_vert.
   static void FormLORVertexCoordinates(FiniteElementSpace &fes_ho,
                                        Vector &X_vert);

   /// Return the vertices of the LOR mesh in E-vector format
   const Vector &GetLORVertexCoordinates() { return X_vert; }

   /// Specialized implementation of SparseIJToCSR for DG spaces.
   void SparseIJToCSR_DG(OperatorHandle &A) const;

protected:
   /// After assembling the "sparse IJ" format, convert it to CSR.
   void SparseIJToCSR(OperatorHandle &A) const;

   /// Assemble the system without eliminating essential DOFs.
   void AssembleWithoutBC(BilinearForm &a, OperatorHandle &A);

   /// @brief Fill in @a sparse_ij and @a sparse_mapping using one of the
   /// specialized LOR assembly kernel classes.
   ///
   /// @sa Specialization classes: BatchedLOR_H1, BatchedLOR_ND, BatchedLOR_RT
   template <typename LOR_KERNEL> void AssemblyKernel(BilinearForm &a);

public:
   /// @name GPU kernel functions
   /// These functions should be considered protected, but they contain
   /// mfem::forall kernels, and so they must be public (this is a compiler
   /// limitation).
   ///@{

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
   /// Assemble the parallel DG matrix (with shared faces).
   void ParAssemble_DG(SparseMatrix &A_local, OperatorHandle &A);

   /// Assemble the system in parallel and place the result in @a A.
   void ParAssemble(BilinearForm &a, const Array<int> &ess_dofs,
                    OperatorHandle &A);
#endif
   ///@}
};

/// @brief Ensure that @a mem has at least capacity @a capacity.
///
/// If the capacity of @a mem is not large enough, delete it and allocate new
/// memory with size @a capacity.
template <typename T>
void EnsureCapacity(Memory<T> &mem, int capacity)
{
   if (mem.Capacity() < capacity)
   {
      mem.Delete();
      mem.New(capacity, mem.GetMemoryType());
   }
}

/// Return the first domain integrator in the form @a i of type @a T.
template <typename T>
static T *GetIntegrator(Array<BilinearFormIntegrator*> *integs)
{
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

template <typename T>
static T *GetIntegrator(BilinearForm &a)
{
   return GetIntegrator<T>(a.GetDBFI());
}

template <typename T>
static T *GetInteriorFaceIntegrator(BilinearForm &a)
{
   return GetIntegrator<T>(a.GetFBFI());
}

/// @brief Return the Gauss-Lobatto rule for geometry @a geom with @a nd1d
/// points per dimension.
IntegrationRule GetLobattoIntRule(Geometry::Type geom, int nd1d);

/// @brief Return the Gauss-Lobatto rule collocated with the element nodes.
///
/// Assumes @a fes uses Gauss-Lobatto basis.
IntegrationRule GetCollocatedIntRule(FiniteElementSpace &fes);

/// @brief Return the Gauss-Lobatto rule collocated with face nodes.
///
/// Assumes @a fes uses Gauss-Lobatto basis.
IntegrationRule GetCollocatedFaceIntRule(FiniteElementSpace &fes);

template <typename INTEGRATOR>
void ProjectLORCoefficient(BilinearForm &a, CoefficientVector &coeff_vector)
{
   INTEGRATOR *i = GetIntegrator<INTEGRATOR>(a);
   if (i)
   {
      // const_cast since Coefficient::Eval is not const...
      auto *coeff = const_cast<Coefficient*>(i->GetCoefficient());
      if (coeff) { coeff_vector.Project(*coeff); }
      else { coeff_vector.SetConstant(1.0); }
   }
   else
   {
      coeff_vector.SetConstant(0.0);
   }
}

/// Abstract base class for the batched LOR assembly kernels.
class BatchedLORKernel
{
protected:
   FiniteElementSpace &fes_ho; ///< The associated high-order space.
   Vector &X_vert; ///< Mesh coordinate vector.
   Vector &sparse_ij; ///< Local element sparsity matrix data.
   Array<int> &sparse_mapping; ///< Local element sparsity pattern.
   IntegrationRule ir; ///< Collocated integration rule.
   QuadratureSpace qs; ///< Quadrature space for coefficients.
   CoefficientVector c1; ///< Coefficient of first integrator.
   CoefficientVector c2; ///< Coefficient of second integrator.
   BatchedLORKernel(FiniteElementSpace &fes_ho_,
                    Vector &X_vert_,
                    Vector &sparse_ij_,
                    Array<int> &sparse_mapping_)
      : fes_ho(fes_ho_), X_vert(X_vert_), sparse_ij(sparse_ij_),
        sparse_mapping(sparse_mapping_), ir(GetCollocatedIntRule(fes_ho)),
        qs(*fes_ho.GetMesh(), ir), c1(qs, CoefficientStorage::COMPRESSED),
        c2(qs, CoefficientStorage::COMPRESSED)
   { }
};

}

#endif
