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
void rGridFuncToQuad1D(
#ifndef __TEMPLATES__
                       const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
#endif
                       const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double* restrict gf,
                       double* restrict out) {
#ifdef __LAMBDA__
  forall(e,numElements,
#else
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#endif
  {
    double r_out[NUM_VDIM][NUM_QUAD_1D];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        r_out[v][qx] = 0;
      }
    }
    for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
      const int gid = l2gMap[(dx) + (NUM_DOFS_1D) * (e)];
      for (int v = 0; v < NUM_VDIM; ++v) {
        const double r_gf = gf[v + gid * NUM_VDIM];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          r_out[v][qx] += r_gf * dofToQuad[(qx) + (NUM_QUAD_1D) * (dx)];
        }
      }
    }
    for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
      for (int v = 0; v < NUM_VDIM; ++v) {
        out[(qx) + (NUM_QUAD_1D) * ((e) + (numElements) * (v))] = r_out[v][qx];
      }
    }
  }
#ifdef __LAMBDA__
         );
#endif
}

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rGridFuncToQuad2D(
#ifndef __TEMPLATES__
                       const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
#endif
                       const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double* restrict gf,
                       double* restrict out) {
#ifdef __LAMBDA__
  forall(e,numElements,
#else
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#endif
  {
    double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          out_xy[v][qy][qx] = 0;
        }
      }
    }
    for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
      double out_x[NUM_VDIM][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          out_x[v][qy] = 0;
        }
      }
      for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
        const int gid = l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)];
        for (int v = 0; v < NUM_VDIM; ++v) {
          const double r_gf = gf[v + gid*NUM_VDIM];
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            out_x[v][qy] += r_gf * dofToQuad[ijN(qy, dx,NUM_QUAD_1D)];
          }
        }
      }
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double d2q = dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xy[v][qy][qx] += d2q * out_x[v][qx];
          }
        }
      }
    }
    for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        for (int v = 0; v < NUM_VDIM; ++v) {
          out[_ijklNM(v, qx, qy, e,NUM_QUAD_1D,numElements)] = out_xy[v][qy][qx];
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
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rGridFuncToQuad3D(
#ifndef __TEMPLATES__
                       const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
#endif
                       const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double* restrict gf,
                       double* restrict out) {
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#else
  forall(e,numElements,
#endif
  {
    double out_xyz[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xyz[v][qz][qy][qx] = 0;
          }
        }
      }
    }
    for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
      double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xy[v][qy][qx] = 0;
          }
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double out_x[NUM_VDIM][NUM_QUAD_1D];
        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_x[v][qx] = 0;
          }
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const int gid = l2gMap[ijklN(dx, dy, dz, e,NUM_DOFS_1D)];
          for (int v = 0; v < NUM_VDIM; ++v) {
            const double r_gf = gf[v + gid*NUM_VDIM];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_x[v][qx] += r_gf * dofToQuad[ijN(qx, dx, NUM_QUAD_1D)];
            }
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = dofToQuad[ijN(qy, dy, NUM_QUAD_1D)];
          for (int v = 0; v < NUM_VDIM; ++v) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_xy[v][qy][qx] += wy * out_x[v][qx];
            }
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = dofToQuad[ijN(qz, dz, NUM_QUAD_1D)];
        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_xyz[v][qz][qy][qx] += wz * out_xy[v][qy][qx];
            }
          }
        }
      }
    }

    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          for (int v = 0; v < NUM_VDIM; ++v) {
            out[_ijklmNM(v, qx, qy, qz, e,NUM_QUAD_1D,numElements)] = out_xyz[v][qz][qy][qx];
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
void rGridFuncToQuad(const int DIM,
                     const int NUM_VDIM,
                     const int NUM_DOFS_1D,
                     const int NUM_QUAD_1D,
                     const int numElements,
                     const double* dofToQuad,
                     const int* l2gMap,
                     const double* gf,
                     double* __restrict out) {
  push(Lime);
#ifndef __LAMBDA__
  const int blck = CUDA_BLOCK_SIZE;
  const int grid = (numElements+blck-1)/blck;
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
    {0x210,&rGridFuncToQuad2D<1,1,2>},
    {0x211,&rGridFuncToQuad2D<1,2,4>},
    {0x212,&rGridFuncToQuad2D<1,3,6>},
    {0x213,&rGridFuncToQuad2D<1,4,8>},
    {0x214,&rGridFuncToQuad2D<1,5,10>},
    {0x215,&rGridFuncToQuad2D<1,6,12>},
    {0x216,&rGridFuncToQuad2D<1,7,14>},
    {0x217,&rGridFuncToQuad2D<1,8,16>},
    {0x218,&rGridFuncToQuad2D<1,9,18>},
    {0x219,&rGridFuncToQuad2D<1,10,20>},
    {0x21A,&rGridFuncToQuad2D<1,11,22>},
    {0x21B,&rGridFuncToQuad2D<1,12,24>},
    {0x21C,&rGridFuncToQuad2D<1,13,26>},
    {0x21D,&rGridFuncToQuad2D<1,14,28>},
    {0x21E,&rGridFuncToQuad2D<1,15,30>},
    {0x21F,&rGridFuncToQuad2D<1,16,32>},

    // 3D
    {0x310,&rGridFuncToQuad3D<1,1,2>},
    {0x311,&rGridFuncToQuad3D<1,2,4>},
    {0x312,&rGridFuncToQuad3D<1,3,6>},
    {0x313,&rGridFuncToQuad3D<1,4,8>},
    {0x314,&rGridFuncToQuad3D<1,5,10>},
    {0x315,&rGridFuncToQuad3D<1,6,12>},
    {0x316,&rGridFuncToQuad3D<1,7,14>},
    {0x317,&rGridFuncToQuad3D<1,8,16>},
    {0x318,&rGridFuncToQuad3D<1,9,18>},
    {0x319,&rGridFuncToQuad3D<1,10,20>},
    {0x31A,&rGridFuncToQuad3D<1,11,22>},
    {0x31B,&rGridFuncToQuad3D<1,12,24>},
    {0x31C,&rGridFuncToQuad3D<1,13,26>},
    {0x31D,&rGridFuncToQuad3D<1,14,28>},
    {0x31E,&rGridFuncToQuad3D<1,15,30>},
    {0x31F,&rGridFuncToQuad3D<1,16,32>},
  };
  if (!call[id]){
    printf("\n[rGridFuncToQuad] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rGridFuncToQuad,id,grid,blck,
        numElements,dofToQuad,l2gMap,gf,out);
#else
  if (DIM==1)
    call0(rGridFuncToQuad1D,id,grid,blck,
          NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,l2gMap,gf,out);
  if (DIM==2)
    call0(rGridFuncToQuad2D,id,grid,blck,
          NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,l2gMap,gf,out);
  if (DIM==3)
    call0(rGridFuncToQuad3D,id,grid,blck,
          NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,l2gMap,gf,out);
#endif
  pop();
}
