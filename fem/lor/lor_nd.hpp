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

#ifndef MFEM_LOR_NEDELEC
#define MFEM_LOR_NEDELEC

#include "lor_batched.hpp"

namespace mfem
{

class BatchedLORNedelec : public BatchedLORAssembly
{
protected:
   double mass_coeff, curl_curl_coeff;
public:
   template <int ORDER> void Assemble2D();
   template <int ORDER> void Assemble3D();
   void AssemblyKernel() override;
   BatchedLORNedelec(BilinearForm &a_,
                     FiniteElementSpace &fes_ho_,
                     const Array<int> &ess_dofs_);
};

}

#endif
