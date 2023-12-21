// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.


#include "mfem.hpp"

#include "tribol/interface/mfem_tribol.hpp"

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   mfem::Mpi::Init();
   int num_ranks = mfem::Mpi::WorldSize();
   int myid = mfem::Mpi::WorldRank();

   return 0;
}