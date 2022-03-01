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

#ifndef MFEM_LOR_DIFFUSION
#define MFEM_LOR_DIFFUSION

#include "lor_batched.hpp"

namespace mfem
{

class BatchedLORDiffusion : public BatchedLORAssembly
{
protected:
   template <int order, bool use_smem=true>
   void AssembleDiffusion(SparseMatrix &A);

   void AssemblyKernel(SparseMatrix &A) override;
public:
   BatchedLORDiffusion(LORBase &lor_disc_,
                       BilinearForm &a_,
                       FiniteElementSpace &fes_ho_,
                       const Array<int> &ess_dofs_);
};

}

#endif
