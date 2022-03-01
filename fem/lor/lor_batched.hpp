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

#ifndef MFEM_LOR_BATCHED
#define MFEM_LOR_BATCHED

#include "lor.hpp"
#include "lor_restriction.hpp"

namespace mfem
{

/// @brief Efficient batched assembly of LOR discretizations on device.
///
/// This class should typically be used by the user-facing classes
/// LORDiscretization and ParLORDiscretization. Only certain bilinear forms are
/// supported, currently:
///
///  - H1 diffusion
class BatchedLORAssembly
{
protected:
   LORBase &lor_disc; ///< Information about the LOR space.
   LORRestriction R; ///< LOR restriction used for sparse matrix assembly.
   FiniteElementSpace &fes_ho; ///< The high-order space.
   const Array<int> &ess_dofs; ///< Essential DOFs to eliminate.

   Vector X_vert; ///< LOR vertex coordinates.

   template <int Q1D> void GetLORVertexCoordinates();

   /// Assemble the system without eliminating essential DOFs.
   SparseMatrix *AssembleWithoutBC();
#ifdef MFEM_USE_MPI
   /// Assemble the system in parallel and place the result in @a A.
   void ParAssemble(OperatorHandle &A);
#endif
   /// Assemble the system, and place the result in @a A.
   void Assemble(OperatorHandle &A);

   /// @brief Pure virtual function for the kernel actually performing the
   /// assembly. Overridden in the derived classes.
   virtual void AssemblyKernel(SparseMatrix &A) = 0;

   /// Called by one of the specialized classes, e.g. BatchedLORDiffusion.
   BatchedLORAssembly(LORBase &lor_disc_,
                      BilinearForm &a_,
                      FiniteElementSpace &fes_ho_,
                      const Array<int> &ess_dofs_);

public:
   /// Does the given form support batched assembly?
   static bool FormIsSupported(BilinearForm &a);
   /// @brief Assembly the given form as a matrix and place the result in @a A.
   ///
   /// In serial, the result will be a SparseMatrix. In parallel, the result
   /// will be a HypreParMatrix.
   static void Assemble(LORBase &lor_disc,
                        BilinearForm &a,
                        FiniteElementSpace &fes_ho,
                        const Array<int> &ess_dofs,
                        OperatorHandle &A);
};

}

#endif
