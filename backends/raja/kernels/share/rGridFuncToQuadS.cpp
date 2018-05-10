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
#include "../raja.hpp"

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rGridFuncToQuad2S(
#ifndef __TEMPLATES__
                       const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
#endif
                       const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double * restrict gf,
                       double* restrict out) {
  const int NUM_QUAD_DOFS_1D = (NUM_QUAD_1D * NUM_DOFS_1D);
  const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;
  // Iterate over elements
#ifdef __LAMBDA__
  forallS(eOff,numElements,M2_ELEMENT_BATCH,
#else
  const int idx = blockIdx.x;
  const int eOff = idx * M2_ELEMENT_BATCH;
  if (eOff < numElements)
#endif
  {
    // Store dof <--> quad mappings
    share double s_dofToQuad[NUM_QUAD_DOFS_1D];//@dim(NUM_QUAD_1D, NUM_DOFS_1D);

    // Store xy planes in shared memory
    share double s_xy[NUM_QUAD_DOFS_1D];//@dim(NUM_DOFS_1D, NUM_QUAD_1D);

    for (int x = 0; x < NUM_MAX_1D; ++x) {
      for (int id = x; id < NUM_QUAD_DOFS_1D; id += NUM_MAX_1D) {
        s_dofToQuad[id] = dofToQuad[id];
      }
    }

    for (int e = eOff; e < (eOff + M2_ELEMENT_BATCH); ++e) {
      if (e < numElements) {
        sync;
#ifdef __LAMBDA__
        for (int dx = 0; dx < NUM_MAX_1D; ++dx) {
#else
        { const int dx = threadIdx.x;
#endif
          if (dx < NUM_DOFS_1D) {
            double r_x[NUM_DOFS_1D];
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              r_x[dy] = gf[l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)]];
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              double xy = 0;
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
                xy += r_x[dy] * s_dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
              }
              s_xy[ijN(dx, qy,NUM_DOFS_1D)] = xy;
            }
          }
        }
        sync;
#ifdef __LAMBDA__
        for (int qy = 0; qy < NUM_MAX_1D; ++qy) {
#else
        { const int qy = threadIdx.x;
#endif
          if (qy < NUM_QUAD_1D) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              double val = 0;
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
                val += s_xy[ijN(dx, qy,NUM_DOFS_1D)] * s_dofToQuad[ijN(qx, dx,NUM_QUAD_1D)];
              }
              out[ijkN(qx, qy, e,NUM_QUAD_1D)] = val;
            }
          }
        }
      }
    }
  }
#ifdef __LAMBDA__
         );
#endif
}
// *****************************************************************************
typedef void (*fGridFuncToQuad)(const int numElements,
                                const double* restrict dofToQuad,
                                const int* restrict l2gMap,
                                const double* gf,
                                double* restrict out);
// *****************************************************************************
void rGridFuncToQuadS(const int DIM,
                     const int NUM_VDIM,
                     const int NUM_DOFS_1D,
                     const int NUM_QUAD_1D,
                     const int numElements,
                     const double* dofToQuad,
                     const int* l2gMap,
                     const double* gf,
                     double* __restrict out) {
  push(Green);
#ifndef __LAMBDA__
  const int grid = ((numElements+M2_ELEMENT_BATCH-1)/M2_ELEMENT_BATCH);
  const int blck = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;
#endif
#ifdef __TEMPLATES__
  const unsigned int id = (DIM<<8)|(NUM_VDIM<<4)|(NUM_DOFS_1D-1);
  assert(LOG2(DIM)<=4);
  assert(LOG2(NUM_VDIM)<=4);
  assert(LOG2(NUM_DOFS_1D-1)<=4);
  assert(NUM_QUAD_1D==2*NUM_DOFS_1D);
  if (NUM_QUAD_1D!=2*NUM_DOFS_1D)
    return exit(printf("\033[31;1m[rGridFuncToQuad] order ERROR: -ok=p -ot=p-1, p in [1,16]\033[m\n"));
  static std::unordered_map<unsigned int, fGridFuncToQuad> call = {
    // 2D
    {0x210,&rGridFuncToQuad2S<1,1,2>},
    {0x211,&rGridFuncToQuad2S<1,2,4>},
    {0x212,&rGridFuncToQuad2S<1,3,6>},
    {0x213,&rGridFuncToQuad2S<1,4,8>},
    {0x214,&rGridFuncToQuad2S<1,5,10>},
    {0x215,&rGridFuncToQuad2S<1,6,12>},
    {0x216,&rGridFuncToQuad2S<1,7,14>},
    {0x217,&rGridFuncToQuad2S<1,8,16>},
    {0x218,&rGridFuncToQuad2S<1,9,18>},
    {0x219,&rGridFuncToQuad2S<1,10,20>},
    {0x21A,&rGridFuncToQuad2S<1,11,22>},
    {0x21B,&rGridFuncToQuad2S<1,12,24>},
    {0x21C,&rGridFuncToQuad2S<1,13,26>},
    {0x21D,&rGridFuncToQuad2S<1,14,28>},
    {0x21E,&rGridFuncToQuad2S<1,15,30>},
    {0x21F,&rGridFuncToQuad2S<1,16,32>},
    // 3D
/*    {0x310,&rGridFuncToQuad3S<1,1,2>},
    {0x311,&rGridFuncToQuad3S<1,2,4>},
    {0x312,&rGridFuncToQuad3S<1,3,6>},
    {0x313,&rGridFuncToQuad3S<1,4,8>},
    {0x314,&rGridFuncToQuad3S<1,5,10>},
    {0x315,&rGridFuncToQuad3S<1,6,12>},
    {0x316,&rGridFuncToQuad3S<1,7,14>},
    {0x317,&rGridFuncToQuad3S<1,8,16>},
    {0x318,&rGridFuncToQuad3S<1,9,18>},
    {0x319,&rGridFuncToQuad3S<1,10,20>},
    {0x31A,&rGridFuncToQuad3S<1,11,22>},
    {0x31B,&rGridFuncToQuad3S<1,12,24>},
    {0x31C,&rGridFuncToQuad3S<1,13,26>},
    {0x31D,&rGridFuncToQuad3S<1,14,28>},
    {0x31E,&rGridFuncToQuad3S<1,15,30>},
    {0x31F,&rGridFuncToQuad3S<1,16,32>},
*/
  };
  if (!call[id]){
    printf("\n[rGridFuncToQuad] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rGridFuncToQuadS,id,grid,blck,
        numElements,dofToQuad,l2gMap,gf,out);
#else
  if (DIM==1) assert(false);
  if (DIM==2)
    call0(rGridFuncToQuad2S,id,grid,blck,
          NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,l2gMap,gf,out);
  if (DIM==3) assert(false);
#endif
  pop();
}
