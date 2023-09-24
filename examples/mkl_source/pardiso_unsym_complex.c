/*******************************************************************************
* Copyright 2004-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*   Content : Intel(R) oneAPI Math Kernel Library (oneMKL) PARDISO C example
*
*******************************************************************************/
/* -------------------------------------------------------------------- */
/*      Example program to show the use of the "PARDISO" routine        */
/*      for complex unsymmetric linear systems                          */
/* -------------------------------------------------------------------- */
/*      This program can be downloaded from the following site:         */
/*      www.pardiso-project.org                                         */
/*                                                                      */
/*  (C) Olaf Schenk, Department of Computer Science,                    */
/*      University of Basel, Switzerland.                               */
/*      Email: olaf.schenk@unibas.ch                                    */
/* -------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"

// Define the format to printf MKL_INT values
#if !defined(MKL_ILP64)
#define IFORMAT "%i"
#else
#define IFORMAT "%lli"
#endif

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64)
#pragma warning(disable:4204)
#endif

int main (void)
{
  /* Matrix data. */

  MKL_INT n = 8;
  MKL_INT ia[9] = { 1, 5, 8, 10, 12, 13, 16, 18, 21 };
  MKL_INT ja[20] =
  { 1,    3,       6, 7,
       2, 3,    5,
          3,             8,
             4,       7,
       2,
          3,       6,    8,
       2,             7,
          3,          7, 8
  };

  MKL_Complex16 a[20] = { {a[0].real = 7.0, a[0].imag = 1.0},
    {a[1].real = 1.0, a[1].imag = 1.0},
    {a[2].real = 2.0, a[2].imag = 1.0},
    {a[3].real = 7.0, a[3].imag = 1.0},
    {a[4].real = -4.0, a[4].imag = 0.0},
    {a[5].real = 8.0, a[5].imag = 1.0},
    {a[6].real = 2.0, a[6].imag = 1.0},
    {a[7].real = 1.0, a[7].imag = 1.0},
    {a[8].real = 5.0, a[8].imag = 1.0},
    {a[9].real = 7.0, a[9].imag = 0.0},
    {a[10].real = 9.0, a[10].imag = 1.0},
    {a[11].real = -4.0, a[11].imag = 1.0},
    {a[12].real = 7.0, a[12].imag = 1.0},
    {a[13].real = 3.0, a[13].imag = 1.0},
    {a[14].real = 8.0, a[14].imag = 0.0},
    {a[15].real = 1.0, a[15].imag = 1.0},
    {a[16].real = 11.0, a[16].imag = 1.0},
    {a[17].real = -3.0, a[17].imag = 1.0},
    {a[18].real = 2.0, a[18].imag = 1.0},
    {a[19].real = 5.0, a[19].imag = 0.0}
  };

  MKL_INT mtype = 13;       /* Real complex unsymmetric matrix */
  // Descriptor of main sparse matrix properties
  struct matrix_descr descrA;
  // Structure with sparse matrix stored in CSR format
  sparse_matrix_t       csrA;
  sparse_operation_t    transA;

  /* RHS and solution vectors. */
  MKL_Complex16 b[8], x[8], bs[8], alpha, beta;
  double res, res0;
  MKL_INT nrhs = 1;     /* Number of right hand sides. */

  /* Internal solver memory pointer pt,                  */
  /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
  /* or void *pt[64] should be OK on both architectures  */
  void *pt[64];

  /* Pardiso control parameters. */
  MKL_INT iparm[64];
  MKL_INT maxfct, mnum, phase, error, msglvl;

  /* Auxiliary variables. */
  MKL_INT i, j;

  MKL_Complex16 ddum;       /* Double dummy */
  MKL_INT idum;         /* Integer dummy. */
/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */
  for (i = 0; i < 64; i++)
    {
      iparm[i] = 0;
    }
  iparm[0] = 1;         /* No solver default */
  iparm[1] = 2;         /* Fill-in reordering from METIS */
  iparm[3] = 0;         /* No iterative-direct algorithm */
  iparm[4] = 0;         /* No user fill-in reducing permutation */
  iparm[5] = 0;         /* Write solution into x */
  iparm[6] = 0;         /* Not in use */
  iparm[7] = 2;         /* Max numbers of iterative refinement steps */
  iparm[8] = 0;         /* Not in use */
  iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
  iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
  iparm[11] = 0;        /* Conjugate transposed/transpose solve */
  iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
  iparm[13] = 0;        /* Output: Number of perturbed pivots */
  iparm[14] = 0;        /* Not in use */
  iparm[15] = 0;        /* Not in use */
  iparm[16] = 0;        /* Not in use */
  iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
  iparm[18] = -1;       /* Output: Mflops for LU factorization */
  iparm[19] = 0;        /* Output: Numbers of CG Iterations */

  maxfct = 1;           /* Maximum number of numerical factorizations.  */
  mnum = 1;         /* Which factorization to use. */

  msglvl = 0;           /* Print statistical information  */
  error = 0;            /* Initialize error flag */

/* -------------------------------------------------------------------- */
/* .. Initialize the internal solver memory pointer. This is only */
/* necessary for the FIRST call of the PARDISO solver. */
/* -------------------------------------------------------------------- */
  for (i = 0; i < 64; i++)
    {
      pt[i] = 0;
    }
/* -------------------------------------------------------------------- */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */
  phase = 11;

  PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
       &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

  if (error != 0)
    {
      printf ("\nERROR during symbolic factorization: " IFORMAT, error);
      exit (1);
    }
  printf ("\nReordering completed ... ");
  printf ("\nNumber of nonzeros in factors  = " IFORMAT, iparm[17]);
  printf ("\nNumber of factorization MFLOPS = " IFORMAT, iparm[18]);

/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */
  phase = 22;

  PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
       &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

  if (error != 0)
    {
      printf ("\nERROR during numerical factorization: " IFORMAT, error);
      exit (2);
    }
  printf ("\nFactorization completed ...\n ");

/* -------------------------------------------------------------------- */
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */
  phase = 33;

  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  descrA.mode = SPARSE_FILL_MODE_UPPER;
  descrA.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_z_create_csr ( &csrA, SPARSE_INDEX_BASE_ONE, n, n, ia, ia+1, ja, a );
  alpha.real = 1.0; alpha.imag = 0.0;
  beta.real  = 0.0; beta.imag  = 0.0;

  /* Set right hand side to one. */
    for (i = 0; i < n; i++)
    {
        b[i].real = 1;
        b[i].imag = 1;
    }
    for (i = 0; i < 3; i++)
    {
        iparm[11] = i;        /* Conjugate transposed/transpose solve */
        if (i == 0)
            transA = SPARSE_OPERATION_NON_TRANSPOSE;
        else if (i == 1)
            transA = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        else
            transA = SPARSE_OPERATION_TRANSPOSE;

      for (j = 0; j < n; j++)
      {
          x[j].real = 0.0;
          x[j].imag = 0.0;
      }
      printf ("\n\nSolving system with iparm[11] = " IFORMAT " ...\n", iparm[11]);
      PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
           &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);

      if (error != 0)
      {
          printf ("\nERROR during solution: " IFORMAT, error);
          exit (3);
      }

      printf ("\nThe solution of the system is: ");
      for (j = 0; j < n; j++)
      {
          printf ("\n x [" IFORMAT "] = % f % f", j, x[j].real, x[j].imag);
      }
      printf ("\n");
// Compute residual
      mkl_sparse_z_mv( transA, alpha, csrA, descrA, x, beta, bs);
      res = 0.0;
      res0 = 0.0;
      for (j = 1; j <= n; j++)
      {
      res += (bs[j - 1].real - b[j - 1].real) * (bs[j - 1].real - b[j - 1].real);
      res += (bs[j - 1].imag - b[j - 1].imag) * (bs[j - 1].imag - b[j - 1].imag);
      res0 += b[j - 1].real * b[j - 1].real;
      res0 += b[j - 1].imag * b[j - 1].imag;
      }
      res = sqrt (res) / sqrt (res0);
      printf ("\nRelative residual = %e", res);
// Check residual
      if (res > 1e-10)
      {
          printf ("Error: residual is too high!\n");
          exit (10 + i);
      }
    }
    mkl_sparse_destroy(csrA);

/* -------------------------------------------------------------------- */
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */
  phase = -1;           /* Release internal memory. */

  PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
       &n, &ddum, ia, ja, &idum, &nrhs,
       iparm, &msglvl, &ddum, &ddum, &error);

  return 0;
}
