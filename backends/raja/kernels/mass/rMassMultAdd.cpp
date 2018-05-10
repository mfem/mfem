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
void rMassMultAdd2D(
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
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#else
  forall(e,numElements,
#endif
  {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0.0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          sol_x[qy] = 0.0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijkN(dx,dy,e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]* s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double d2q = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += d2q * sol_x[qx];
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] *= oper[ijkN(qx,qy,e,NUM_QUAD_1D)];
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xy[qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijkN(dx,dy,e,NUM_DOFS_1D)] += q2d * sol_x[dx];
          }
        }
      }
  }
#ifdef __LAMBDA__
  );
#endif
}

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rMassMultAdd3D(
#ifndef __TEMPLATES__
                    const int NUM_DOFS_1D,
                    const int NUM_QUAD_1D,
#endif
                    const int numElements,
                    const double* dofToQuad,
                    const double* dofToQuadD,
                    const double* quadToDof,
                    const double* quadToDofD,
                    const double* oper,
                    const double* solIn,
                    double* __restrict solOut) {
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#else
  forall(e,numElements,
#endif
  {
    double sol_xyz[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xyz[qz][qy][qx] = 0;
        }
      }
    }
    for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_x[qx] = 0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += wy * sol_x[qx];
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
          }
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,e,NUM_QUAD_1D)];
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      double sol_xy[NUM_DOFS_1D][NUM_DOFS_1D];
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_xy[dy][dx] = 0;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xyz[qz][qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_xy[dy][dx] += wy * sol_x[dx];
          }
        }
      }
      for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
        const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
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
void rMassMultAdd(const int DIM,
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
  push(Lime);
#ifndef __LAMBDA__
  const int blck = 256;
  const int grid = (numElements+blck-1)/blck;
#endif
#ifdef __TEMPLATES__
  assert(LOG2(DIM)<=4);
  assert((NUM_QUAD_1D&1)==0);
  assert(LOG2(NUM_DOFS_1D-1)<=8);
  assert(LOG2(NUM_QUAD_1D>>1)<=8);
  const unsigned int id = (DIM<<16)|((NUM_DOFS_1D-1)<<8)|(NUM_QUAD_1D>>1);
  static std::unordered_map<unsigned int, fMassMultAdd> call = {
    // 2D
    {0x20001,&rMassMultAdd2D<1,2>},    {0x20101,&rMassMultAdd2D<2,2>},
    {0x20102,&rMassMultAdd2D<2,4>},    {0x20202,&rMassMultAdd2D<3,4>},
    {0x20203,&rMassMultAdd2D<3,6>},    {0x20303,&rMassMultAdd2D<4,6>},
    {0x20304,&rMassMultAdd2D<4,8>},    {0x20404,&rMassMultAdd2D<5,8>},
    {0x20405,&rMassMultAdd2D<5,10>},   {0x20505,&rMassMultAdd2D<6,10>},
    {0x20506,&rMassMultAdd2D<6,12>},   {0x20606,&rMassMultAdd2D<7,12>},
    {0x20607,&rMassMultAdd2D<7,14>},   {0x20707,&rMassMultAdd2D<8,14>},
    {0x20708,&rMassMultAdd2D<8,16>},   {0x20808,&rMassMultAdd2D<9,16>},
    {0x20809,&rMassMultAdd2D<9,18>},   {0x20909,&rMassMultAdd2D<10,18>},
    {0x2090A,&rMassMultAdd2D<10,20>},  {0x20A0A,&rMassMultAdd2D<11,20>},
    {0x20A0B,&rMassMultAdd2D<11,22>},  {0x20B0B,&rMassMultAdd2D<12,22>},
    {0x20B0C,&rMassMultAdd2D<12,24>},  {0x20C0C,&rMassMultAdd2D<13,24>},
    {0x20C0D,&rMassMultAdd2D<13,26>},  {0x20D0D,&rMassMultAdd2D<14,26>},
    {0x20D0E,&rMassMultAdd2D<14,28>},  {0x20E0E,&rMassMultAdd2D<15,28>},
    {0x20E0F,&rMassMultAdd2D<15,30>},  {0x20F0F,&rMassMultAdd2D<16,30>},
    {0x20F10,&rMassMultAdd2D<16,32>},  {0x21010,&rMassMultAdd2D<17,32>},
    // 3D
    {0x30001,&rMassMultAdd3D<1,2>},    {0x30101,&rMassMultAdd3D<2,2>},
    {0x30102,&rMassMultAdd3D<2,4>},    {0x30202,&rMassMultAdd3D<3,4>},
    {0x30203,&rMassMultAdd3D<3,6>},    {0x30303,&rMassMultAdd3D<4,6>},
    {0x30304,&rMassMultAdd3D<4,8>},    {0x30404,&rMassMultAdd3D<5,8>},
    {0x30405,&rMassMultAdd3D<5,10>},   {0x30505,&rMassMultAdd3D<6,10>},
    {0x30506,&rMassMultAdd3D<6,12>},   {0x30606,&rMassMultAdd3D<7,12>},
    {0x30607,&rMassMultAdd3D<7,14>},   {0x30707,&rMassMultAdd3D<8,14>},
    {0x30708,&rMassMultAdd3D<8,16>},   {0x30808,&rMassMultAdd3D<9,16>},
    {0x30809,&rMassMultAdd3D<9,18>},   {0x30909,&rMassMultAdd3D<10,18>},
    {0x3090A,&rMassMultAdd3D<10,20>},  {0x30A0A,&rMassMultAdd3D<11,20>},
    {0x30A0B,&rMassMultAdd3D<11,22>},  {0x30B0B,&rMassMultAdd3D<12,22>},
    {0x30B0C,&rMassMultAdd3D<12,24>},  {0x30C0C,&rMassMultAdd3D<13,24>},
    {0x30C0D,&rMassMultAdd3D<13,26>},  {0x30D0D,&rMassMultAdd3D<14,26>},
    {0x30D0E,&rMassMultAdd3D<14,28>},  {0x30E0E,&rMassMultAdd3D<15,28>},
    {0x30E0F,&rMassMultAdd3D<15,30>},  {0x30F0F,&rMassMultAdd3D<16,30>},
    {0x30F10,&rMassMultAdd3D<16,32>},  {0x31010,&rMassMultAdd3D<17,32>},
  };
  if(!call[id]){
    printf("\n[rMassMultAdd] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rMassMultAdd,id,grid,blck,
        numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
#else
  if (DIM==1) assert(false);
  if (DIM==2)
    call0(rMassMultAdd2D,id,grid,blck,
          NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
  if (DIM==3)
    call0(rMassMultAdd3D,id,grid,blck,
          NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
#endif
  pop();
}
