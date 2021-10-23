// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BLOCKBILINEARFORM
#define MFEM_BLOCKBILINEARFORM

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"

namespace mfem
{

/** @brief A "square matrix" operator for the associated FE space and
    BLFIntegrators The sum of all the BLFIntegrators can be used form the matrix
    M.  */
class BlockBilinearForm : public Matrix
{

protected:
   /// Sparse matrix \f$ M \f$ to be associated with the form. Owned.
   SparseMatrix *mat;

   /** @brief Sparse Matrix \f$ M_e \f$ used to store the eliminations
        from the b.c.  Owned.
       \f$ M + M_e = M_{original} \f$ */
   SparseMatrix *mat_e;

   /// FE spaces on which the block form lives. Not owned.
   Array<FiniteElementSpace * > fespaces;

   /** @brief Indicates the Mesh::sequence corresponding to the current state of
      the BilinearForm. */
   long sequence;

   /** @brief Indicates the BilinearFormIntegrator%s stored in #domain_integs,
    #boundary_integs, #interior_face_integs, and #boundary_face_integs are
    owned by another BilinearForm. */
   int extern_bfs;

   /// Set of Domain Integrators to be applied.
   Array<BlockBilinearFormIntegrator*> domain_integs;

   DenseMatrix elemmat;
   Array<int>  vdofs;

   DenseTensor *element_matrices; ///< Owned.

   /** This data member allows one to specify what should be done to the
    diagonal matrix entries and corresponding RHS values upon elimination of
    the constrained DoFs. */
   DiagonalPolicy diag_policy;

   // Allocate appropriate SparseMatrix and assign it to mat
   void AllocMat();

   void ConformingAssemble();


private:

public:

   /// Creates bilinear form associated with FE spaces @a *fespaces.
   BlockBilinearForm(Array<FiniteElementSpace * > & fespaces);

   /// Get the size of the BilinearForm as a square matrix.
   int Size() const { return height; }


   /// Pre-allocate the internal SparseMatrix before assembly.
   void AllocateMatrix() { if (mat == NULL) { AllocMat(); } }

   /// Returns a reference to: \f$ M_{ij} \f$
   const double &operator()(int i, int j) { return (*mat)(i,j); }


   /// Matrix vector multiplication:  \f$ y = M x \f$
   virtual void Mult(const Vector &x, Vector &y) const;

   /** @brief Matrix vector multiplication with the original uneliminated
       matrix.  The original matrix is \f$ M + M_e \f$ so we have:
       \f$ y = M x + M_e x \f$ */
   void FullMult(const Vector &x, Vector &y) const
   { mat->Mult(x, y); mat_e->AddMult(x, y); }

   virtual double &Elem(int i, int j);
   virtual const double &Elem(int i, int j) const;
   virtual MatrixInverse *Inverse() const;

   /// Finalizes the matrix initialization.
   virtual void Finalize(int skip_zeros = 1);

   /// Returns a const reference to the sparse matrix.
   const SparseMatrix &SpMat() const
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return *mat;
   }

   /// Returns a reference to the sparse matrix:  \f$ M \f$
   SparseMatrix &SpMat()
   {
      MFEM_VERIFY(mat, "mat is NULL and can't be dereferenced");
      return *mat;
   }

   /// Returns a const reference to the sparse matrix of eliminated b.c.: \f$ M_e \f$
   const SparseMatrix &SpMatElim() const
   {
      MFEM_VERIFY(mat_e, "mat_e is NULL and can't be dereferenced");
      return *mat_e;
   }

   /// Returns a reference to the sparse matrix of eliminated b.c.: \f$ M_e \f$
   SparseMatrix &SpMatElim()
   {
      MFEM_VERIFY(mat_e, "mat_e is NULL and can't be dereferenced");
      return *mat_e;
   }

   /// Adds new Domain Integrator. Assumes ownership of @a bfi.
   void AddDomainIntegrator(BlockBilinearFormIntegrator *bfi);

   /// Sets all sparse values of \f$ M \f$ and \f$ M_e \f$ to 'a'.
   void operator=(const double a)
   {
      if (mat != NULL) { *mat = a; }
      if (mat_e != NULL) { *mat_e = a; }
   }

   /// Assembles the form i.e. sums over all domain integrators.
   void Assemble(int skip_zeros = 1);

   void ComputeElementMatrices();

   /// Free the memory used by the element matrices.
   void FreeElementMatrices()
   { delete element_matrices; element_matrices = NULL; }

   /// Compute the element matrix of the given element
   /** The element matrix is computed by calling the domain integrators
       or the one stored internally by a prior call of ComputeElementMatrices()
       is returned when available.
   */
   void ComputeElementMatrix(int i, DenseMatrix &elmat);

   /// Destroys bilinear form.
   virtual ~BlockBilinearForm();

};

} // namespace mfem

#endif
