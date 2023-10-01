// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "pmortarassembler.hpp"
#endif // MFEM_USE_MPI
#endif // MFEM_USE_MOONOLITH

namespace mfem
{

/*!
 * @brief Initializes the par_moonolith library. It also calls MPI_Init.
 * @param argc Standard argument passed to the main function.
 * @param argv Standard argument passed to the main function.
 */
void InitTransfer(int argc, char *argv[]);

/*!
 * @brief Finalize the par_moonolith library.
 * @return Zero if everything has succeeded.
 */
int FinalizeTransfer();

#ifdef MFEM_USE_MPI
/*!
 * @brief Initializes the transfer library. It does not call MPI_Init, but uses
 * the communicator defined by the user. This method can be called only after
 * MPI_Init.
 * @param argc Standard argument passed to the main function.
 * @param argv Standard argument passed to the main function.
 * @param comm The user defined communicator.
 */
void InitTransfer(int argc, char *argv[], MPI_Comm comm);
#endif

} // namespace mfem

#endif // MFEM_TRANSFER
