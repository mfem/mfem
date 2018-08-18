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
#include "../kernels.hpp"

// *****************************************************************************
extern "C" kernel
void rIniGeom2D(const int, const int, const int, const double*, const double*,
                double*, double*, double*);
extern "C" kernel
void rIniGeom3D(const int, const int, const int, const double*, const double*,
                double*, double*, double*);

// *****************************************************************************
void rIniGeom(const int DIM,
              const int NUM_DOFS,
              const int NUM_QUAD,
              const int numElements,
              const double* dofToQuadD,
              const double* nodes,
              double* J,
              double* invJ,
              double* detJ)
{
  push();
#ifndef __LAMBDA__
  const int blck = CUDA_BLOCK_SIZE;
  const int grid = (numElements+blck-1)/blck;
#endif
  if (DIM==2)
    call0(rIniGeom2D,id,grid,blck,NUM_DOFS,NUM_QUAD,
          numElements,dofToQuadD,nodes,J,invJ,detJ);
  if (DIM==3)
    call0(rIniGeom3D,id,grid,blck,NUM_DOFS,NUM_QUAD,
          numElements,dofToQuadD,nodes,J,invJ,detJ);
  assert(DIM==2 || DIM==3);
  pop();
}
