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
kernel
#endif
void rNodeCopyByVDim0(const int elements,
                      const int numDofs,
                      const int ndofs,
                      const int dims,
                      const int* eMap,
                      const double* Sx,
                      double* nodes){
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < elements)
#else
  push(Lime);
  forall(e,elements,
#endif
  {
    for(int dof = 0; dof < numDofs; ++dof) {
      const int lid = dof+numDofs*e;
      const int gid = eMap[lid];
      for(int v = 0; v < dims; ++v) {
        const int moffset = v+dims*lid;
        const int voffset = gid+v*ndofs;
        nodes[moffset] = Sx[voffset];
      }
    }
  }
#ifdef __LAMBDA__
         );
  pop();
#endif
}

// *****************************************************************************
void rNodeCopyByVDim(const int elements,
                     const int numDofs,
                     const int ndofs,
                     const int dims,
                     const int* eMap,
                     const double* Sx,
                     double* nodes){
#ifndef __LAMBDA__
  cuKer(rNodeCopyByVDim,elements,numDofs,ndofs,dims,eMap,Sx,nodes);
#else
  rNodeCopyByVDim0(elements,numDofs,ndofs,dims,eMap,Sx,nodes);
#endif
}


// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS,
         const int NUM_QUAD> kernel
#endif
void rIniGeom1D(
#ifndef __TEMPLATES__
                const int NUM_DOFS,
                const int NUM_QUAD,
#endif
                const int numElements,
                const double* restrict dofToQuadD,
                const double* restrict nodes,
                double* restrict J,
                double* restrict invJ,
                double* restrict detJ) {
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#else
  forall(e,numElements,
#endif
  {
    double s_nodes[NUM_DOFS];
    for (int q = 0; q < NUM_QUAD; ++q) {
      for (int d = q; d < NUM_DOFS; d += NUM_QUAD) { 
        s_nodes[d] = nodes[ijkN(0,d,e,NUM_QUAD)];
      }
    }
    for (int q = 0; q < NUM_QUAD; ++q) {
      double J11 = 0;
      for (int d = 0; d < NUM_DOFS; ++d) {
        const double wx = dofToQuadD[ijN(q,d,NUM_DOFS)];
        J11 += wx * s_nodes[d];
      } 
      J[ijN(q,e,NUM_QUAD)] = J11;
      invJ[ijN(q, e,NUM_QUAD)] = 1.0 / J11;
      detJ[ijN(q, e,NUM_QUAD)] = J11;
    }
  }
#ifdef __LAMBDA__
          );
#endif
}

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS,
         const int NUM_QUAD> kernel
#endif
void rIniGeom2D(
#ifndef __TEMPLATES__
                const int NUM_DOFS,
                const int NUM_QUAD,
#endif
                const int numElements,
                const double* restrict dofToQuadD,
                const double* restrict nodes,
                double* restrict J,
                double* restrict invJ,
                double* restrict detJ) {
#ifndef __LAMBDA__
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < numElements)
#else
  forall(el,numElements,
#endif
  {
    double s_nodes[2 * NUM_DOFS];
    for (int q = 0; q < NUM_QUAD; ++q) {
      for (int d = q; d < NUM_DOFS; d +=NUM_QUAD) {
        s_nodes[ijN(0,d,2)] = nodes[ijkNM(0,d,el,2,NUM_DOFS)];
        s_nodes[ijN(1,d,2)] = nodes[ijkNM(1,d,el,2,NUM_DOFS)];
      }
    }
    for (int q = 0; q < NUM_QUAD; ++q) {
      double J11 = 0; double J12 = 0;
      double J21 = 0; double J22 = 0;
      for (int d = 0; d < NUM_DOFS; ++d) {
        const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
        const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
        const double x = s_nodes[ijN(0,d,2)];
        const double y = s_nodes[ijN(1,d,2)]; 
        J11 += (wx * x); J12 += (wx * y); 
        J21 += (wy * x); J22 += (wy * y);
      }
      const double r_detJ = (J11 * J22)-(J12 * J21);
      J[ijklNM(0, 0, q, el,2,NUM_QUAD)] = J11;
      J[ijklNM(1, 0, q, el,2,NUM_QUAD)] = J12;
      J[ijklNM(0, 1, q, el,2,NUM_QUAD)] = J21;
      J[ijklNM(1, 1, q, el,2,NUM_QUAD)] = J22;
      const double r_idetJ = 1.0 / r_detJ;
      invJ[ijklNM(0, 0, q, el,2,NUM_QUAD)] =  J22 * r_idetJ;
      invJ[ijklNM(1, 0, q, el,2,NUM_QUAD)] = -J12 * r_idetJ;
      invJ[ijklNM(0, 1, q, el,2,NUM_QUAD)] = -J21 * r_idetJ;
      invJ[ijklNM(1, 1, q, el,2,NUM_QUAD)] =  J11 * r_idetJ;
      detJ[ijN(q, el,NUM_QUAD)] = r_detJ;
    }
  }
#ifdef __LAMBDA__
          );
#endif
}

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS,
         const int NUM_QUAD> kernel
#endif
void rIniGeom3D(
#ifndef __TEMPLATES__
                const int NUM_DOFS,
                const int NUM_QUAD,
#endif
                const int numElements,
                const double* restrict dofToQuadD,
                const double* restrict nodes,
                double* restrict J,
                double* restrict invJ,
                double* restrict detJ) {
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#else
  forall(e,numElements,
#endif
  {
    double s_nodes[3*NUM_DOFS];
    for (int q = 0; q < NUM_QUAD; ++q) {
      for (int d = q; d < NUM_DOFS; d += NUM_QUAD) {
        s_nodes[ijN(0,d,3)] = nodes[ijkNM(0, d, e,3,NUM_DOFS)];
        s_nodes[ijN(1,d,3)] = nodes[ijkNM(1, d, e,3,NUM_DOFS)];
        s_nodes[ijN(2,d,3)] = nodes[ijkNM(2, d, e,3,NUM_DOFS)];
      }
    }
    for (int q = 0; q < NUM_QUAD; ++q) {
      double J11 = 0; double J12 = 0; double J13 = 0;
      double J21 = 0; double J22 = 0; double J23 = 0;
      double J31 = 0; double J32 = 0; double J33 = 0;
      for (int d = 0; d < NUM_DOFS; ++d) {
        const double wx = dofToQuadD[ijkNM(0, q, d,3,NUM_QUAD)];
        const double wy = dofToQuadD[ijkNM(1, q, d,3,NUM_QUAD)];
        const double wz = dofToQuadD[ijkNM(2, q, d,3,NUM_QUAD)];
        const double x = s_nodes[ijN(0, d,3)];
        const double y = s_nodes[ijN(1, d,3)];
        const double z = s_nodes[ijN(2, d,3)];
        J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
        J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
        J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
      }
      const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                             (J13 * J21 * J32) -
                             (J13 * J22 * J31)-(J12 * J21 * J33)-(J11 * J23 * J32));
      J[ijklNM(0, 0, q, e,3,NUM_QUAD)] = J11;
      J[ijklNM(1, 0, q, e,3,NUM_QUAD)] = J12;
      J[ijklNM(2, 0, q, e,3,NUM_QUAD)] = J13;
      J[ijklNM(0, 1, q, e,3,NUM_QUAD)] = J21;
      J[ijklNM(1, 1, q, e,3,NUM_QUAD)] = J22;
      J[ijklNM(2, 1, q, e,3,NUM_QUAD)] = J23;
      J[ijklNM(0, 2, q, e,3,NUM_QUAD)] = J31;
      J[ijklNM(1, 2, q, e,3,NUM_QUAD)] = J32;
      J[ijklNM(2, 2, q, e,3,NUM_QUAD)] = J33;

      const double r_idetJ = 1.0 / r_detJ;
      invJ[ijklNM(0, 0, q, e,3,NUM_QUAD)] = r_idetJ * ((J22 * J33)-(J23 * J32));
      invJ[ijklNM(1, 0, q, e,3,NUM_QUAD)] = r_idetJ * ((J32 * J13)-(J33 * J12));
      invJ[ijklNM(2, 0, q, e,3,NUM_QUAD)] = r_idetJ * ((J12 * J23)-(J13 * J22));

      invJ[ijklNM(0, 1, q, e,3,NUM_QUAD)] = r_idetJ * ((J23 * J31)-(J21 * J33));
      invJ[ijklNM(1, 1, q, e,3,NUM_QUAD)] = r_idetJ * ((J33 * J11)-(J31 * J13));
      invJ[ijklNM(2, 1, q, e,3,NUM_QUAD)] = r_idetJ * ((J13 * J21)-(J11 * J23));

      invJ[ijklNM(0, 2, q, e,3,NUM_QUAD)] = r_idetJ * ((J21 * J32)-(J22 * J31));
      invJ[ijklNM(1, 2, q, e,3,NUM_QUAD)] = r_idetJ * ((J31 * J12)-(J32 * J11));
      invJ[ijklNM(2, 2, q, e,3,NUM_QUAD)] = r_idetJ * ((J11 * J22)-(J12 * J21));
      detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
    }
  }
#ifdef __LAMBDA__
          );
#endif
}

// *****************************************************************************
typedef void (*fIniGeom)(const int numElements,
                         const double* restrict dofToQuadD,
                         const double* restrict nodes,
                         double* restrict J,
                         double* restrict invJ,
                         double* restrict detJ);


// *****************************************************************************
void rIniGeom(const int DIM,
              const int NUM_DOFS,
              const int NUM_QUAD,
              const int numElements,
              const double* dofToQuadD,
              const double* nodes,
              double* restrict J,
              double* restrict invJ,
              double* restrict detJ) {
  push(Lime);
#ifndef __LAMBDA__
  const int blck = CUDA_BLOCK_SIZE;
  const int grid = (numElements+blck-1)/blck;
#endif
#ifdef __TEMPLATES__
  const unsigned int dofs1D = IROOT(DIM,NUM_DOFS);
  const unsigned int quad1D = IROOT(DIM,NUM_QUAD);
  const unsigned int id = (DIM<<4)|(dofs1D-2);
  assert(LOG2(DIM)<=4);
  assert(LOG2(dofs1D-2)<=4);
  if (quad1D!=2*(dofs1D-1))
    return exit(printf("\033[31;1m[rIniGeom] order ERROR: -ok=p -ot=p-1, p in [1,16] (%d,%d)\033[m\n",quad1D,dofs1D));
  assert(quad1D==2*(dofs1D-1));
  static std::unordered_map<unsigned int, fIniGeom> call = {
    // 2D
    {0x20,&rIniGeom2D<2*2,(2*2-2)*(2*2-2)>},
    {0x21,&rIniGeom2D<3*3,(3*2-2)*(3*2-2)>},
    {0x22,&rIniGeom2D<4*4,(4*2-2)*(4*2-2)>},
    {0x23,&rIniGeom2D<5*5,(5*2-2)*(5*2-2)>},
    {0x24,&rIniGeom2D<6*6,(6*2-2)*(6*2-2)>},
    {0x25,&rIniGeom2D<7*7,(7*2-2)*(7*2-2)>},
    {0x26,&rIniGeom2D<8*8,(8*2-2)*(8*2-2)>},
    {0x27,&rIniGeom2D<9*9,(9*2-2)*(9*2-2)>},
    {0x28,&rIniGeom2D<10*10,(10*2-2)*(10*2-2)>},
    {0x29,&rIniGeom2D<11*11,(11*2-2)*(11*2-2)>},
    {0x2A,&rIniGeom2D<12*12,(12*2-2)*(12*2-2)>},
    {0x2B,&rIniGeom2D<13*13,(13*2-2)*(13*2-2)>},
    {0x2C,&rIniGeom2D<14*14,(14*2-2)*(14*2-2)>},
    {0x2D,&rIniGeom2D<15*15,(15*2-2)*(15*2-2)>},
    {0x2E,&rIniGeom2D<16*16,(16*2-2)*(16*2-2)>},
    {0x2F,&rIniGeom2D<17*17,(17*2-2)*(17*2-2)>},
    // 3D
    {0x30,&rIniGeom3D<2*2*2,2*2*2>},
    {0x31,&rIniGeom3D<3*3*3,4*4*4>},
    {0x32,&rIniGeom3D<4*4*4,6*6*6>},
    {0x33,&rIniGeom3D<5*5*5,8*8*8>},
    {0x34,&rIniGeom3D<6*6*6,10*10*10>},
    {0x35,&rIniGeom3D<7*7*7,12*12*12>},
    {0x36,&rIniGeom3D<8*8*8,14*14*14>},
    {0x37,&rIniGeom3D<9*9*9,16*16*16>},
    {0x38,&rIniGeom3D<10*10*10,18*18*18>},
    {0x39,&rIniGeom3D<11*11*11,20*20*20>},
    {0x3A,&rIniGeom3D<12*12*12,22*22*22>},
    {0x3B,&rIniGeom3D<13*13*13,24*24*24>},
    {0x3C,&rIniGeom3D<14*14*14,26*26*26>},
    {0x3D,&rIniGeom3D<15*15*15,28*28*28>},
    {0x3E,&rIniGeom3D<16*16*16,30*30*30>},
    {0x3F,&rIniGeom3D<17*17*17,32*32*32>},
  };
  if (!call[id]){
    printf("\n[rIniGeom] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rIniGeom2D,id,grid,blck,
        numElements,dofToQuadD,nodes,J,invJ,detJ);
#else
  if (DIM==2)
    call0(rIniGeom2D,id,grid,blck,NUM_DOFS,NUM_QUAD,
          numElements,dofToQuadD,nodes,J,invJ,detJ);
  if (DIM==3)
    call0(rIniGeom3D,id,grid,blck,NUM_DOFS,NUM_QUAD,
          numElements,dofToQuadD,nodes,J,invJ,detJ);
  assert(DIM==2 || DIM==3);
#endif
  pop();
}
