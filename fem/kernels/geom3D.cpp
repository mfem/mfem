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

#include "../../general/okina.hpp"

// *****************************************************************************
template<const int NUM_DOFS,
         const int NUM_QUAD> __kernel__
void rIniGeom3D(const int numElements,
                const double* __restrict dofToQuadD,
                const double* __restrict nodes,
                double* __restrict J,
                double* __restrict invJ,
                double* __restrict detJ){
#ifdef __NVCC__
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
#else
   forall(e,numElements,
#endif
   {
      double s_nodes[3*NUM_DOFS];
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         for (int d = q; d < NUM_DOFS; d += NUM_QUAD)
         {
            s_nodes[ijN(0,d,3)] = nodes[ijkNM(0, d, e,3,NUM_DOFS)];
            s_nodes[ijN(1,d,3)] = nodes[ijkNM(1, d, e,3,NUM_DOFS)];
            s_nodes[ijN(2,d,3)] = nodes[ijkNM(2, d, e,3,NUM_DOFS)];
         }
      }
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double J11 = 0; double J12 = 0; double J13 = 0;
         double J21 = 0; double J22 = 0; double J23 = 0;
         double J31 = 0; double J32 = 0; double J33 = 0;
         for (int d = 0; d < NUM_DOFS; ++d)
         {
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
#ifndef __NVCC__
           );
#endif
}
/*
template __kernel__ void rIniGeom3D<8,8>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<27,64>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<64,216>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<125,512>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<216,1000>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<343,1728>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<512,2744>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<729,4096>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<1000,5832>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<1331,8000>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<1728,10648>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<2197,13824>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<2744,17576>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<3375,21952>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<4096,27000>(int, double const*, double const*, double*, double*, double*);
template __kernel__ void rIniGeom3D<4913,32768>(int, double const*, double const*, double*, double*, double*);
*/
