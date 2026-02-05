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

#ifndef MFEM_LOR_NEDELEC
#define MFEM_LOR_NEDELEC

#include "lor_batched.hpp"

namespace mfem
{

// BatchedLORKernel specialization for Nedelec spaces. Not user facing. See the
// classes BatchedLORAssembly and BatchedLORKernel .
class BatchedLOR_ND : BatchedLORKernel
{
public:
   template <int ORDER, int SDIM> void Assemble2D();
   template <int ORDER> void Assemble3D();
   BatchedLOR_ND(BilinearForm &a,
                 FiniteElementSpace &fes_ho_,
                 Vector &X_vert_,
                 Vector &sparse_ij_,
                 Array<int> &sparse_mapping_)
      : BatchedLORKernel(fes_ho_, X_vert_, sparse_ij_, sparse_mapping_)
   {
      ProjectLORCoefficient<VectorFEMassIntegrator>(a, c1);
      ProjectLORCoefficient<CurlCurlIntegrator>(a, c2);
   }
};

}

#include "lor_nd_impl.hpp"

#endif
