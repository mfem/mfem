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

#include "blockstaticcond.hpp"

namespace mfem
{


BlockStaticCondensation::BlockStaticCondensation(Array<FiniteElementSpace *> &
                                                 fes_)
{

}

void BlockStaticCondensation::SetSpaces()
{

}

void BlockStaticCondensation::Init()
{

}

void BlockStaticCondensation::AssembleMatrix(int el, const DenseMatrix &elmat)
{

}

void BlockStaticCondensation::Finalize()
{

}

void BlockStaticCondensation::SetEssentialTrueDofs(const Array<int>
                                                   &ess_tdof_list)
{

}

void BlockStaticCondensation::EliminateReducedTrueDofs(const Array<int>
                                                       &ess_rtdof_list,
                                                       Matrix::DiagonalPolicy dpolicy)
{

}

void BlockStaticCondensation::EliminateReducedTrueDofs(Matrix::DiagonalPolicy
                                                       dpolicy)
{

}

void BlockStaticCondensation::ReduceRHS(const Vector &b, Vector &sc_b) const
{

}

void BlockStaticCondensation::ReduceSolution(const Vector &sol,
                                             Vector &sc_sol) const
{

}

void BlockStaticCondensation::ReduceSystem(Vector &x, Vector &b, Vector &X,
                                           Vector &B,
                                           int copy_interior) const
{

}

void BlockStaticCondensation::ConvertMarkerToReducedTrueDofs(
   const Array<int> &ess_tdof_marker,
   Array<int> &ess_rtdof_marker) const
{

}

void BlockStaticCondensation::ConvertListToReducedTrueDofs(
   const Array<int> &ess_tdof_list,
   Array<int> &ess_rtdof_list) const
{

}

void BlockStaticCondensation::ComputeSolution(const Vector &b,
                                              const Vector &sc_sol,
                                              Vector &sol) const
{

}

}