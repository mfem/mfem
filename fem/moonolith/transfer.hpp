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

#ifndef MFEM_TRANSFER
#define MFEM_TRANSFER

#include "../fem.hpp"

#ifdef MFEM_USE_MOONOLITH
#include "mortarassembler.hpp"
#ifdef MFEM_USE_MPI
#include "parallel/pmortarassembler.hpp"
#endif // MFEM_USE_MPI
#endif // MFEM_USE_MOONOLITH

namespace mfem
{

void InitTransfer(int argc, char *argv[]);
int FinalizeTransfer();

#ifdef MFEM_USE_MPI
void InitTransfer(int argc, char *argv[], MPI_Comm comm);
#endif

} // namespace mfem

#endif // MFEM_TRANSFER
