# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Defines the following variables:
#   - MKL_CPARDISO_FOUND
#   - MKL_CPARDISO_LIBRARIES
#   - MKL_CPARDISO_INCLUDE_DIRS

if(NOT MKL_MPI_WRAPPER_LIB)
  message(FATAL_ERROR "MKL CPardiso enabled but no MKL MPI Wrapper lib specified")
endif()

if(NOT MKL_LIBRARY_DIR)
  message(WARNING "Using default MKL library path. Double check the variable MKL_LIBRARY_DIR")
  set(MKL_LIBRARY_DIR "lib")
endif()

include(MfemCmakeUtilities)
mfem_find_package(MKL_CPARDISO MKL_CPARDISO
    MKL_CPARDISO_DIR "include" mkl_cluster_sparse_solver.h ${MKL_LIBRARY_DIR} mkl_core
  "Paths to headers required by MKL CPardiso." "Libraries required by MKL CPARDISO."
  ADD_COMPONENT MKL_LP64 "include" "" ${MKL_LIBRARY_DIR} mkl_intel_lp64
  ADD_COMPONENT MKL_SEQUENTIAL "include" "" ${MKL_LIBRARY_DIR} mkl_sequential
  ADD_COMPONENT MKL_MPI_WRAPPER "include" "" ${MKL_LIBRARY_DIR} ${MKL_MPI_WRAPPER_LIB}
  CHECK_BUILD MKL_CPARDISO_VERSION_OK TRUE
  "
#include <mpi.h>
#include <mkl.h>
#include <mkl_cluster_sparse_solver.h>
int main (void)
{
    MKL_INT n = 5;
    MKL_INT ia[6]  = { 1, 4, 6, 9, 12, 14};
    MKL_INT ja[13] = {  1, 2, 4,        /* index of non-zeros in 1 row*/
                        1, 2,           /* index of non-zeros in 2 row*/
                        3, 4, 5,        /* index of non-zeros in 3 row*/
                        1, 3, 4,        /* index of non-zeros in 4 row*/
                        2, 5            /* index of non-zeros in 5 row*/
    };
    double a[13] = {
                         1.0, -1.0, /*0*/ -3.0, /*0*/
                        -2.0,  5.0, /*0*/ /*0*/ /*0*/
                        /*0*/  4.0,  6.0,  4.0, /*0*/
                        -4.0, /*0*/  2.0,  7.0, /*0*/
                        /*0*/  8.0, /*0*/ /*0*/ -5.0
                    };

    MKL_INT mtype = 11; /* set matrix type to \"real unsymmetric matrix\" */
    MKL_INT nrhs  = 1;  /* Number of right hand sides. */
    double b[5], x[5], bs[5], res, res0; /* RHS and solution vectors. */

    /* Internal solver memory pointer pt
     *       32-bit:      int pt[64] or void *pt[64];
     *       64-bit: long int pt[64] or void *pt[64]; */
    void *pt[64] = { 0 };

    /* Cluster Sparse Solver control parameters. */
    MKL_INT iparm[64] = { 0 };
    MKL_INT maxfct, mnum, phase, msglvl, error;

    /* Auxiliary variables. */
    double  ddum; /* Double dummy   */
    MKL_INT idum; /* Integer dummy. */
    MKL_INT i, j;
    int     mpi_stat = 0;
    int     argc = 0;
    int     comm, rank;
    char*   uplo;
    char**  argv;

    mpi_stat = MPI_Init( &argc, &argv );
    mpi_stat = MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    comm =  MPI_Comm_c2f( MPI_COMM_WORLD );

    iparm[ 0] =  1; /* Solver default parameters overriden with provided by iparm */
    iparm[ 1] =  2; /* Use METIS for fill-in reordering */
    iparm[ 5] =  0; /* Write solution into x */
    iparm[ 7] =  2; /* Max number of iterative refinement steps */
    iparm[ 9] = 13; /* Perturb the pivot elements with 1E-13 */
    iparm[10] =  1; /* Use nonsymmetric permutation and scaling MPS */
    iparm[12] =  1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
    iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1; /* Output: Mflops for LU factorization */
    iparm[26] =  1; /* Check input data for correctness */
    iparm[39] =  0; /* Input: matrix/rhs/solution stored on master */
    maxfct = 1; /* Maximum number of numerical factorizations. */
    mnum   = 1; /* Which factorization to use. */
    msglvl = 1; /* Print statistical information in file */
    error  = 0; /* Initialize error flag */

    phase = 11;
    cluster_sparse_solver ( pt, &maxfct, &mnum, &mtype, &phase,
                &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &comm, &error );

    mpi_stat = MPI_Finalize();
    return error;
}
")
