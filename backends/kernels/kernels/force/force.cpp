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
void rForceMult2D(const int, const int, const int, const int,
                  const int, const int, const double*,
                  const double*, const double*, const double*, const double*,
                  double* );

void rForceMultTranspose2D(const int NUM_DIM,
                           const int NUM_DOFS_1D,
                           const int NUM_QUAD_1D,
                           const int L2_DOFS_1D,
                           const int H1_DOFS_1D,
                           const int numElements,
                           const double* __restrict__ L2QuadToDof,
                           const double* __restrict__ H1DofToQuad,
                           const double* __restrict__ H1DofToQuadD,
                           const double* __restrict__ stressJinvT,
                           const double* __restrict__ v,
                           double* __restrict__ e);

void rForceMult3D(const int NUM_DIM,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int L2_DOFS_1D,
                  const int H1_DOFS_1D,
                  const int numElements,
                  const double* __restrict__ L2DofToQuad,
                  const double* __restrict__ H1QuadToDof,
                  const double* __restrict__ H1QuadToDofD,
                  const double* __restrict__ stressJinvT,
                  const double* __restrict__ e,
                  double* __restrict__ v);

void rForceMultTranspose3D(const int NUM_DIM,
                           const int NUM_DOFS_1D,
                           const int NUM_QUAD_1D,
                           const int L2_DOFS_1D,
                           const int H1_DOFS_1D,
                           const int numElements,
                           const double* __restrict__ L2QuadToDof,
                           const double* __restrict__ H1DofToQuad,
                           const double* __restrict__ H1DofToQuadD,
                           const double* __restrict__ stressJinvT,
                           const double* __restrict__ v,
                           double* __restrict__ e);
// *****************************************************************************
void rForceMult(const int NUM_DIM,
                const int NUM_DOFS_1D,
                const int NUM_QUAD_1D,
                const int L2_DOFS_1D,
                const int H1_DOFS_1D,
                const int nzones,
                const double* L2QuadToDof,
                const double* H1DofToQuad,
                const double* H1DofToQuadD,
                const double* stressJinvT,
                const double* e,
                double* v)
{
   push(Lime);
#ifndef __LAMBDA__
   const int blck = CUDA_BLOCK_SIZE;
   const int grid = (nzones+blck-1)/blck;
#endif
   if (NUM_DIM==2)
     call0(rForceMult2D,id,grid,blck,
           NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
           nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
   if (NUM_DIM==3)
     call0(rForceMult3D,id,grid,blck,
           NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
           nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
   if (NUM_DIM!=2 && NUM_DIM!=3)
   {
     exit(printf("\n[rForceMult] NUM_DIM!=2 && NUM_DIM!=3 ERROR"));
   }
   pop();
}

// *****************************************************************************
void rForceMultTranspose(const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
                         const int nzones,
                         const double* L2QuadToDof,
                         const double* H1DofToQuad,
                         const double* H1DofToQuadD,
                         const double* stressJinvT,
                         const double* v,
                         double* e)
{
   push(Lime);
#ifndef __LAMBDA__
   const int blck = CUDA_BLOCK_SIZE;
   const int grid = (nzones+blck-1)/blck;
#endif
   if (NUM_DIM==2)
     call0(rForceMultTranspose2D,id,grid,blck,
           NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
           nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
   if (NUM_DIM==3)
     call0(rForceMultTranspose3D,id,grid,blck,
           NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
           nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
   if (NUM_DIM!=2 && NUM_DIM!=3)
   {
     exit(printf("\n[rForceMultTranspose] NUM_DIM!=2 && NUM_DIM!=3 ERROR"));
   }
   pop();
}

