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
#ifndef MFEM_KERNELS_MASS
#define MFEM_KERNELS_MASS

// *****************************************************************************
void rMassMultAdd(const int dim,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int numElements,
                  const double* restrict dofToQuad,
                  const double* restrict dofToQuadD,
                  const double* restrict quadToDof,
                  const double* restrict quadToDofD,
                  const double* restrict op,
                  const double* restrict x,
                  double* restrict y);

// *****************************************************************************
void rMassMultAddS(const int dim,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int numElements,
                  const double* restrict dofToQuad,
                  const double* restrict dofToQuadD,
                  const double* restrict quadToDof,
                  const double* restrict quadToDofD,
                  const double* restrict op,
                  const double* restrict x,
                  double* restrict y);

#endif // MFEM_KERNELS_MASS
