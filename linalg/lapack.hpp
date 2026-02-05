// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LAPACK_HPP
#define MFEM_LAPACK_HPP

#include "../config/config.hpp"

#ifdef MFEM_USE_SINGLE
#define MFEM_LAPACK_PREFIX(stub) s##stub
#define MFEM_LAPACK_COMPLEX(stub) c##stub
#elif defined(MFEM_USE_DOUBLE)
#define MFEM_LAPACK_PREFIX(stub) d##stub
#define MFEM_LAPACK_COMPLEX(stub) z##stub
#endif

#ifdef MFEM_USE_LAPACK

#include <complex>

namespace mfem
{

extern "C" void
MFEM_LAPACK_PREFIX(gemm_)(char *, char *, int *, int *, int *, real_t *,
                          real_t *, int *, real_t *, int *, real_t *, real_t *,
                          int *);
extern "C" void
MFEM_LAPACK_PREFIX(getrf_)(int *, int *, real_t *, int *, int *, int *);
extern "C" void
MFEM_LAPACK_PREFIX(getrs_)(char *, int *, int *, real_t *, int *, int *,
                           real_t *, int *, int *);
extern "C" void
MFEM_LAPACK_PREFIX(getri_)(int *N, real_t *A, int *LDA, int *IPIV, real_t *WORK,
                           int *LWORK, int *INFO);
extern "C" void
MFEM_LAPACK_PREFIX(gbsv_)(int *, int *, int *, int *, real_t *, int *, int *,
                          real_t *, int *, int *);
extern "C" void
MFEM_LAPACK_PREFIX(gbtrs_)(char *, int *, int *, int *, int *, real_t *, int *,
                           int *, real_t *, int *, int *);

extern "C" void
MFEM_LAPACK_PREFIX(syevr_)(char *JOBZ, char *RANGE, char *UPLO, int *N,
                           real_t *A, int *LDA, real_t *VL, real_t *VU, int *IL,
                           int *IU, real_t *ABSTOL, int *M, real_t *W,
                           real_t *Z, int *LDZ, int *ISUPPZ, real_t *WORK,
                           int *LWORK, int *IWORK, int *LIWORK, int *INFO);
extern "C" void
MFEM_LAPACK_PREFIX(syev_)(char *JOBZ, char *UPLO, int *N, real_t *A, int *LDA,
                          real_t *W, real_t *WORK, int *LWORK, int *INFO);
extern "C" void
MFEM_LAPACK_PREFIX(sygv_) (int *ITYPE, char *JOBZ, char *UPLO, int * N,
                           real_t *A, int *LDA, real_t *B, int *LDB, real_t *W,
                           real_t *WORK, int *LWORK, int *INFO);
extern "C" void
MFEM_LAPACK_PREFIX(gesvd_)(char *JOBU, char *JOBVT, int *M, int *N, real_t *A,
                           int *LDA, real_t *S, real_t *U, int *LDU, real_t *VT,
                           int *LDVT, real_t *WORK, int *LWORK, int *INFO);
extern "C" void
MFEM_LAPACK_PREFIX(trsm_)(char *side, char *uplo, char *transa, char *diag,
                          int *m, int *n, real_t *alpha, real_t *a, int *lda,
                          real_t *b, int *ldb);
extern "C" void
MFEM_LAPACK_PREFIX(ggev_)(char *jobvl, char *jobvr, int *n, real_t *a, int *lda,
                          real_t *B, int *ldb, real_t *alphar, real_t *alphai,
                          real_t *beta, real_t *vl, int * ldvl, real_t * vr,
                          int * ldvr, real_t * work, int * lwork, int* info);

// Cholesky factorizations/solves
extern "C" void
MFEM_LAPACK_PREFIX(potrf_)(char *, int *, real_t *, int *, int *);
// Solve
extern "C" void
MFEM_LAPACK_PREFIX(potrs_)(char *, int *, int *, real_t *, int *, real_t *,
                           int *, int *);
// Triangular Solves
extern "C" void
MFEM_LAPACK_PREFIX(trtrs_)(char *, char*, char *, int *, int *, real_t *, int *,
                           real_t *, int *, int *);
extern "C" void
MFEM_LAPACK_PREFIX(potri_)(char *, int *, real_t *, int*, int *);

// LAPACK routines for NNLSSolver
extern "C" void
MFEM_LAPACK_PREFIX(ormqr_)(char *, char *, int *, int *, int *, real_t *, int*,
                           real_t *, real_t *, int *, real_t *, int*, int*);

extern "C" void
MFEM_LAPACK_PREFIX(geqrf_)(int *, int *, real_t *, int *, real_t *, real_t *,
                           int *, int *);

extern "C" void
MFEM_LAPACK_PREFIX(gemv_)(char *, int *, int *, real_t *, real_t *, int *,
                          real_t *, int *, real_t *, real_t *, int *);

extern "C" void
MFEM_LAPACK_PREFIX(trsm_)(char *side, char *uplo, char *transa, char *diag,
                          int *m, int *n, real_t *alpha, real_t *a, int *lda,
                          real_t *b, int *ldb);

// Complex
extern "C" void
MFEM_LAPACK_COMPLEX(getrf_)(int *, int *, std::complex<real_t> *, int *, int *,
                            int *);
extern "C" void
MFEM_LAPACK_COMPLEX(getrs_)(char *, int *, int *, std::complex<real_t> *, int *,
                            int *, std::complex<real_t> *, int *, int *);
extern "C" void
MFEM_LAPACK_COMPLEX(getri_)(int *, std::complex<real_t> *, int *, int *,
                            std::complex<real_t> *, int *, int *);
extern "C" void
MFEM_LAPACK_COMPLEX(trsm_)(char *, char *, char *, char *, int *, int *,
                           std::complex<real_t> *, std::complex<real_t> *,
                           int *, std::complex<real_t> *, int *);
extern "C" void
MFEM_LAPACK_COMPLEX(potrf_)(char *, int *, std::complex<real_t> *, int *,
                            int *);

extern "C" void
MFEM_LAPACK_COMPLEX(trtrs_)(char *, char*, char *, int *, int *,
                            std::complex<real_t> *, int *,
                            std::complex<real_t> *, int *, int *);
extern "C" void
MFEM_LAPACK_COMPLEX(potri_)(char *, int *, std::complex<real_t> *, int*, int *);

extern "C" void
MFEM_LAPACK_COMPLEX(potrs_)(char *, int *, int *, std::complex<real_t> *, int *,
                            std::complex<real_t> *, int *, int *);

}

#endif

#endif
