// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include "../config/config.hpp"
#include "globals.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace mfem
{

OutStream out(std::cout);
OutStream err(std::cerr);


std::string MakeParFilename(const std::string &prefix, const int myid,
                            const std::string suffix, const int width)
{
   std::stringstream fname;
   fname << prefix << std::setw(width) << std::setfill('0') << myid << suffix;
   return fname.str();
}

#ifdef MFEM_COUNT_FLOPS
namespace internal
{
long long flop_count;
}
#endif

#ifdef MFEM_USE_MPI

MPI_Comm MFEM_COMM_WORLD = MPI_COMM_WORLD;

MPI_Comm GetGlobalMPI_Comm()
{
   return MFEM_COMM_WORLD;
}

void SetGlobalMPI_Comm(MPI_Comm comm)
{
   MFEM_COMM_WORLD = comm;
}

#endif

}
