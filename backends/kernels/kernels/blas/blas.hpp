// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef MFEM_KERNELS_BLAS
#define MFEM_KERNELS_BLAS

// **** BLAS1 ******************************************************************
void vector_neg(const int, double*);
void vector_op_eq(const int, const double, double*);
void vector_xpay(const int, const double, double*, const double*,const double*);
void vector_xsy(const int, double*, const double*, const double*);
void vector_axpy(const int, const double, double*, const double*);
void vector_axpby(const int, const double,const double, double*, const double*);
void vector_axpby3(const int, const double,const double, double*, const double*, const double*);
void vector_map_dofs(const int, double*, const double*, const int*);
template <class T>
void vector_map_add_dofs(const int, T*, const T*, const int*);
void vector_clear_dofs(const int, double*, const int*);
void vector_vec_sub(const int, double*, const double*);
void vector_vec_add(const int, double*, const double*);
void vector_vec_mul(const int, double*, const double);
void vector_set_subvector(const int, double*, const double*,
                          const int*);
void vector_get_subvector(const int, double*, const double*,
                          const int*);
void vector_set_subvector_const(const int, const double, double*,
                                const int*);
double vector_dot(const int, const double*, const double*);
double vector_min(const int, const double*);

#endif // MFEM_KERNELS_BLAS
