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
#ifndef MFEM_KERNELS_GEOM
#define MFEM_KERNELS_GEOM

// *****************************************************************************
void rNodeCopyByVDim(const int elements,
                     const int numDofs,
                     const int ndofs,
                     const int dims,
                     const int* eMap,
                     const double* Sx,
                     double* nodes);

// *****************************************************************************
void rIniGeom(const int dim,
              const int nDofs,
              const int nQuads,
              const int nzones,
              const double* dofToQuadD,
              const double* nodes,
              double* J,
              double* invJ,
              double* detJ);

#endif // MFEM_KERNELS_GEOM
