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

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "mfem.hpp"

#ifndef MFEM_USE_MPI
#error This test assumes that mpi is available
#endif

#include <mpi.h>

int main( int argc, char** argv )
{
   MPI_Init( &argc, &argv );

   int result = Catch::Session().run( argc, argv );

   MPI_Finalize();

   return result;
}