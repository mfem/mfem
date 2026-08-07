/**
 * test_opt_mma_parallel.cpp  —  Parallel MMA -> OptimizationSolver bridge.
 * Runs the shared distributed compliance scenarios (inequality, equality, and
 * zero-rank) with MMAOptimizationSolverParallel. See test_par_common.hpp.
 */

#include "MMAOptSolver.hpp"
#include "test_par_common.hpp"
#include <cstdio>

#ifndef MFEM_USE_MPI
int main() { printf("MFEM built without MPI — skipping.\n"); return 0; }
#else
int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   const int gfail = optpar::RunAllScenarios<mfem_mma::MMAOptimizationSolverParallel>(
                        MPI_COMM_WORLD, "MMA parallel");
   MPI_Finalize();
   return gfail > 0 ? 1 : 0;
}
#endif
