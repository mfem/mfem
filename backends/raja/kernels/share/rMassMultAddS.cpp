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
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rMassMultAdd2S(
#ifndef __TEMPLATES__
                    const int NUM_DOFS_1D,
                    const int NUM_QUAD_1D,
#endif
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
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
    share double s_quadToDof[NUM_QUAD_DOFS_1D];//@dim(NUM_DOFS_1D, NUM_QUAD_1D);

    // Store xy planes in shared memory
    share double s_xy[NUM_QUAD_DOFS_1D];//@dim(NUM_DOFS_1D, NUM_QUAD_1D);
    share double s_xy2[NUM_QUAD_2D];//@dim(NUM_QUAD_1D, NUM_QUAD_1D);

    double r_x[NUM_MAX_1D];

#ifdef __LAMBDA__
    for (int x = 0; x < NUM_MAX_1D; ++x/*;inner*/)
#else
      const int x = threadIdx.x;
#endif
    {
      for (int id = x; id < NUM_QUAD_DOFS_1D; id += NUM_MAX_1D) {
        s_dofToQuad[id]  = dofToQuad[id];
        s_quadToDof[id]  = quadToDof[id];
      }
    }

    for (int e = eOff; e < (eOff + M2_ELEMENT_BATCH); ++e) {
      if (e < numElements) {
        sync;
#ifdef __LAMBDA__
        for (int dx = 0; dx < NUM_MAX_1D; ++dx) {
#else
        {
          const int dx = threadIdx.x;
#endif
        
          if (dx < NUM_DOFS_1D) {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              s_xy[ijN(dx, qy,NUM_DOFS_1D)] = 0;
            }
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              r_x[dy] = solIn[ijkN(dx, dy, e,NUM_DOFS_1D)];
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
        for (int qy = 0; qy < NUM_MAX_1D; ++qy){
#else
        {
          const int qy = threadIdx.x;
#endif
          if (qy < NUM_QUAD_1D) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              double s = 0;
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
                s += s_xy[ijN(dx, qy,NUM_DOFS_1D)] * s_dofToQuad[ijN(qx, dx,NUM_QUAD_1D)];
              }
              s_xy2[ijN(qx, qy,NUM_QUAD_1D)] = s * oper[ijkN(qx, qy, e,NUM_QUAD_1D)];
            }
          }
        }
        sync;
#ifdef __LAMBDA__
        for (int qx = 0; qx < NUM_MAX_1D; ++qx){
#else
        {
          const int qx = threadIdx.x;
#endif
          if (qx < NUM_QUAD_1D) {
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              s_xy[ijN(dy, qx,NUM_DOFS_1D)] = 0;
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              r_x[qy] = s_xy2[ijN(qx, qy,NUM_QUAD_1D)];
            }
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              double s = 0;
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                s += r_x[qy] * s_quadToDof[ijN(dy, qy,NUM_DOFS_1D)];
              }
              s_xy[ijN(dy, qx,NUM_DOFS_1D)] = s;
            }
          }
        }
        sync;
#ifdef __LAMBDA__
        for (int dx = 0; dx < NUM_MAX_1D; ++dx){
#else
        {
            const int dx = threadIdx.x;
#endif
            if (dx < NUM_DOFS_1D) {
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              double s = 0;
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                s += (s_xy[ijN(dy, qx,NUM_DOFS_1D)] * s_quadToDof[ijN(dx, qx,NUM_DOFS_1D)]);
              }
              solOut[ijkN(dx, dy, e,NUM_DOFS_1D)] += s;
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
typedef void (*fMassMultAdd)(const int numElements,
                             const double* dofToQuad,
                             const double* dofToQuadD,
                             const double* quadToDof,
                             const double* quadToDofD,
                             const double* oper,
                             const double* solIn,
                             double* __restrict solOut);

// *****************************************************************************
void rMassMultAddS(const int DIM,
                   const int NUM_DOFS_1D,
                   const int NUM_QUAD_1D,
                   const int numElements,
                   const double* dofToQuad,
                   const double* dofToQuadD,
                   const double* quadToDof,
                   const double* quadToDofD,
                   const double* op,
                   const double* x,
                   double* __restrict y) {
  push(Green);
#ifndef __LAMBDA__
  const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;
  const int grid = ((numElements+M2_ELEMENT_BATCH-1)/M2_ELEMENT_BATCH);
  const int blck = NUM_MAX_1D;
#endif
#ifdef __TEMPLATES__
  assert(LOG2(DIM)<=4);
  assert((NUM_QUAD_1D&1)==0);
  assert(LOG2(NUM_DOFS_1D-1)<=8);
  assert(LOG2(NUM_QUAD_1D>>1)<=8);
  const unsigned int id = (DIM<<16)|((NUM_DOFS_1D-1)<<8)|(NUM_QUAD_1D>>1);
  static std::unordered_map<unsigned int, fMassMultAdd> call = {
    // 2D
    {0x20001,&rMassMultAdd2S<1,2>},    {0x20101,&rMassMultAdd2S<2,2>},
    {0x20102,&rMassMultAdd2S<2,4>},    {0x20202,&rMassMultAdd2S<3,4>},
    {0x20203,&rMassMultAdd2S<3,6>},    {0x20303,&rMassMultAdd2S<4,6>},
    {0x20304,&rMassMultAdd2S<4,8>},    {0x20404,&rMassMultAdd2S<5,8>},
    {0x20405,&rMassMultAdd2S<5,10>},   {0x20505,&rMassMultAdd2S<6,10>},
    {0x20506,&rMassMultAdd2S<6,12>},   {0x20606,&rMassMultAdd2S<7,12>},
    {0x20607,&rMassMultAdd2S<7,14>},   {0x20707,&rMassMultAdd2S<8,14>},
    {0x20708,&rMassMultAdd2S<8,16>},   {0x20808,&rMassMultAdd2S<9,16>},
    {0x20809,&rMassMultAdd2S<9,18>},   {0x20909,&rMassMultAdd2S<10,18>},
    {0x2090A,&rMassMultAdd2S<10,20>},  {0x20A0A,&rMassMultAdd2S<11,20>},
    {0x20A0B,&rMassMultAdd2S<11,22>},  {0x20B0B,&rMassMultAdd2S<12,22>},
    {0x20B0C,&rMassMultAdd2S<12,24>},  {0x20C0C,&rMassMultAdd2S<13,24>},
    {0x20C0D,&rMassMultAdd2S<13,26>},  {0x20D0D,&rMassMultAdd2S<14,26>},
    {0x20D0E,&rMassMultAdd2S<14,28>},  {0x20E0E,&rMassMultAdd2S<15,28>},
    {0x20E0F,&rMassMultAdd2S<15,30>},  {0x20F0F,&rMassMultAdd2S<16,30>},
    {0x20F10,&rMassMultAdd2S<16,32>},  {0x21010,&rMassMultAdd2S<17,32>},
    // 3D
/*
    {0x30001,&rMassMultAdd3S<1,2>},    {0x30101,&rMassMultAdd3S<2,2>},
    {0x30102,&rMassMultAdd3S<2,4>},    {0x30202,&rMassMultAdd3S<3,4>},
    {0x30203,&rMassMultAdd3S<3,6>},    {0x30303,&rMassMultAdd3S<4,6>},
    {0x30304,&rMassMultAdd3S<4,8>},    {0x30404,&rMassMultAdd3S<5,8>},
    {0x30405,&rMassMultAdd3S<5,10>},   {0x30505,&rMassMultAdd3S<6,10>},
    {0x30506,&rMassMultAdd3S<6,12>},   {0x30606,&rMassMultAdd3S<7,12>},
    {0x30607,&rMassMultAdd3S<7,14>},   {0x30707,&rMassMultAdd3S<8,14>},
    {0x30708,&rMassMultAdd3S<8,16>},   {0x30808,&rMassMultAdd3S<9,16>},
    {0x30809,&rMassMultAdd3S<9,18>},   {0x30909,&rMassMultAdd3S<10,18>},
    {0x3090A,&rMassMultAdd3S<10,20>},  {0x30A0A,&rMassMultAdd3S<11,20>},
    {0x30A0B,&rMassMultAdd3S<11,22>},  {0x30B0B,&rMassMultAdd3S<12,22>},
    {0x30B0C,&rMassMultAdd3S<12,24>},  {0x30C0C,&rMassMultAdd3S<13,24>},
    {0x30C0D,&rMassMultAdd3S<13,26>},  {0x30D0D,&rMassMultAdd3S<14,26>},
    {0x30D0E,&rMassMultAdd3S<14,28>},  {0x30E0E,&rMassMultAdd3S<15,28>},
    {0x30E0F,&rMassMultAdd3S<15,30>},  {0x30F0F,&rMassMultAdd3S<16,30>},
    {0x30F10,&rMassMultAdd3S<16,32>},  {0x31010,&rMassMultAdd3S<17,32>},
*/
  };
  if(!call[id]){
    printf("\n[rMassMultAddS] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rMassMultAdd2S,id,grid,blck,
        numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
#else
  if (DIM==1) assert(false);
  if (DIM==2)
    call0(rMassMultAdd2S,id,grid,blck,
          NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y); 
  if (DIM==3) assert(false);
#endif
  pop();
}
