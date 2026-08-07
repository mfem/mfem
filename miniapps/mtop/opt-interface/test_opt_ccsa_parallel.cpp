/**
 * test_opt_ccsa_parallel.cpp  —  Parallel CCSA (Bregman) -> OptimizationSolver.
 * Runs the shared distributed compliance scenarios (inequality, equality, and
 * zero-rank) with CCSAOptimizationSolverParallel. See test_par_common.hpp.
 */

#include "CCSAOptSolver.hpp"
#include "test_par_common.hpp"
#include <cstdio>

#ifndef MFEM_USE_MPI
int main() { printf("MFEM built without MPI — skipping.\n"); return 0; }
#else
int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   const int gfail = optpar::RunAllScenarios<mfem_mma::CCSAOptimizationSolverParallel>(
                        MPI_COMM_WORLD, "CCSA parallel");
   MPI_Finalize();
   return gfail > 0 ? 1 : 0;
}
#endif
