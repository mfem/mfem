// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

// Instantiates the 3D Navier-Stokes operator. Kept in its own translation unit
// so the q-function kernels for each dimension build in parallel; the
// definitions are in navier_solver.hpp.

#include "navier_solver.hpp"

namespace mfem
{
namespace dfem_navier
{

template class NavierStokesOperator<3>;

} // namespace dfem_navier
} // namespace mfem
