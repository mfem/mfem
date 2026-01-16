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

#ifndef MFEM_TEST_SAME_MATRICES_HPP
#define MFEM_TEST_SAME_MATRICES_HPP

#include "mfem.hpp"
#include "unit_tests.hpp"

namespace mfem
{

#ifndef MFEM_USE_MPI
#define HYPRE_BigInt int
#endif // MFEM_USE_MPI

inline void TestSameMatrices(
   SparseMatrix &A1, const SparseMatrix &A2,
   HYPRE_BigInt *cmap1=nullptr,
   std::unordered_map<HYPRE_BigInt,int> *cmap2inv=nullptr)
{
   REQUIRE(A1.Height() == A2.Height());
   int n = A1.Height();

   const int *I1 = A1.HostReadI();
   const int *J1 = A1.HostReadJ();
   const real_t *V1 = A1.HostReadData();

   A2.HostReadI();
   A2.HostReadJ();
   A2.HostReadData();

   real_t error = 0.0;

   for (int i=0; i<n; ++i)
   {
      for (int jj=I1[i]; jj<I1[i+1]; ++jj)
      {
         int j = J1[jj];
         if (cmap1)
         {
            if (cmap2inv->count(cmap1[j]) > 0)
            {
               j = (*cmap2inv)[cmap1[j]];
            }
            else
            {
               error = std::max(error, std::fabs(V1[jj]));
               continue;
            }
         }
         error = std::max(error, std::fabs(V1[jj] - A2(i,j)));
      }
   }

   REQUIRE(error == MFEM_Approx(0.0, 1e-10));
}

#ifdef MFEM_USE_MPI

inline void TestSameMatrices(HypreParMatrix &A1, const HypreParMatrix &A2)
{
   HYPRE_BigInt *cmap1, *cmap2;
   SparseMatrix diag1, offd1, diag2, offd2;

   A1.GetDiag(diag1);
   A2.GetDiag(diag2);
   A1.GetOffd(offd1, cmap1);
   A2.GetOffd(offd2, cmap2);

   TestSameMatrices(diag1, diag2);

   if (cmap1)
   {
      std::unordered_map<HYPRE_BigInt,int> cmap2inv;
      for (int i=0; i<offd2.Width(); ++i) { cmap2inv[cmap2[i]] = i; }
      TestSameMatrices(offd1, offd2, cmap1, &cmap2inv);
   }
   else
   {
      TestSameMatrices(offd1, offd2);
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem

#endif
