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
#ifdef __TEMPLATES__
template<const int NUM_DOFS,
         const int NUM_QUAD> kernel
#endif
void rIniGeom2D(
#ifndef __TEMPLATES__
                const int,const int,
#endif
                const int,const double*,const double*,
                double*,double*,double*);

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS,
         const int NUM_QUAD> kernel
#endif
void rIniGeom3D(
#ifndef __TEMPLATES__
                const int,const int,
#endif
                const int,const double*,const double*,
                double*,double*,double*);

// *****************************************************************************
typedef void (*fIniGeom)(const int, const double*, const double*,
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
              double* detJ){
   push();
#ifndef __LAMBDA__
   const int blck = CUDA_BLOCK_SIZE;
   const int grid = (numElements+blck-1)/blck;
#endif
#ifdef __TEMPLATES__
   dbg("__TEMPLATES__");
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
