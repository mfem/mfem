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

#include "transfer.hpp"

#include "par_moonolith_instance.hpp"

namespace mfem
{

void InitTransfer(int argc, char *argv[])
{
   moonolith::Moonolith::Init(argc, argv);
}
int FinalizeTransfer() { return moonolith::Moonolith::Finalize(); }

#ifdef MFEM_USE_MPI
void InitTransfer(int argc, char *argv[], MPI_Comm comm)
{
   moonolith::Moonolith::Init(argc, argv, comm);
}
#endif

} // namespace mfem