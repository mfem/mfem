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

#include "filteredsolver.hpp"


namespace mfem
{

FilteredSolver::FilteredSolver(MPI_Comm comm_)
{

}

FilteredSolver::FilteredSolver(const Operator &op)
{

}

void FilteredSolver::SetOperator(const Operator &op)
{

}

void FilteredSolver::Mult(const Vector &x, Vector &y) const
{

}

void FilteredSolver::SetPrintLevel(int print_lvl)
{
   print_level = print_lvl;
}

FilteredSolver::~FilteredSolver()
{

}

void FilteredSolver::Init(MPI_Comm comm_)
{}

} // namespace mfem