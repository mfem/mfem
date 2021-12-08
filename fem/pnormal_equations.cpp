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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParNormalEquations::pAllocMat()
{
   // mat = new BlockMatrix(dof_offsets);
   // mat->owns_blocks = 1;

   // for (int i = 0; i<mat->NumRowBlocks(); i++)
   // {
   //    int h = dof_offsets[i+1] - dof_offsets[i];
   //    for (int j = 0; j<mat->NumColBlocks(); j++)
   //    {
   //       int w = dof_offsets[j+1] - dof_offsets[j];
   //       mat->SetBlock(i,j,new SparseMatrix(h, w));
   //    }
   // }
   // y = new BlockVector(dof_offsets);
   // *y = 0.;
}

// void ParNormalEquations::Finalize(int skip_zeros)
// {

// }






// void ParNormalEquations::BuildProlongation()
// {

// }

// void ParNormalEquations::FormLinearSystem(const Array<int>
//                                        &ess_tdof_list,
//                                        Vector &x,
//                                        OperatorHandle &A, Vector &X,
//                                        Vector &B, int copy_interior)
// {

// }

// void ParNormalEquations::FormSystemMatrix(const Array<int>
//                                        &ess_tdof_list,
//                                        OperatorHandle &A)
// {

// }

// void ParNormalEquations::EliminateVDofsInRHS(
//    const Array<int> &vdofs, const Vector &x, Vector &b)
// {

// }

// void ParNormalEquations::EliminateVDofs(const Array<int> &vdofs,
//                                      Operator::DiagonalPolicy dpolicy)
// {

// }

// void ParNormalEquations::RecoverFEMSolution(const Vector &X,
//                                          Vector &x)
// {
// }

// ParNormalEquations::~NormalEquations()
// {


// }

} // namespace mfem

#endif
