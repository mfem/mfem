#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_integrators.hpp"

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   MPI_Finalize();
   return 0;
}
