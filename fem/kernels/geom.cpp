// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"
#include "../../general/okina.hpp"

namespace mfem
{
namespace kernels
{
namespace fem
{

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> static
void Geom2D(const int numElements,
            const double* __restrict dofToQuadD,
            const double* __restrict nodes,
            double* __restrict J,
            double* __restrict invJ,
            double* __restrict detJ)
{
   const int NUM_DOFS = NUM_DOFS_1D*NUM_DOFS_1D;
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D;
   MFEM_FORALL(e, numElements,
   {
      double s_nodes[2 * NUM_DOFS_1D * NUM_DOFS_1D];
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         for (int d = q; d < NUM_DOFS; d +=NUM_QUAD)
         {
            s_nodes[ijN(0,d,2)] = nodes[ijkNM(0,d,e,2,NUM_DOFS)];
            s_nodes[ijN(1,d,2)] = nodes[ijkNM(1,d,e,2,NUM_DOFS)];
         }
      }
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double J11 = 0; double J12 = 0;
         double J21 = 0; double J22 = 0;
         for (int d = 0; d < NUM_DOFS; ++d)
         {
            const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
            const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
            const double x = s_nodes[ijN(0,d,2)];
            const double y = s_nodes[ijN(1,d,2)];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
         }
         const double r_detJ = (J11 * J22)-(J12 * J21);
         assert(r_detJ!=0.0);
         J[ijklNM(0,0,q,e,2,NUM_QUAD)] = J11;
         J[ijklNM(1,0,q,e,2,NUM_QUAD)] = J12;
         J[ijklNM(0,1,q,e,2,NUM_QUAD)] = J21;
         J[ijklNM(1,1,q,e,2,NUM_QUAD)] = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ[ijklNM(0,0,q,e,2,NUM_QUAD)] =  J22 * r_idetJ;
         invJ[ijklNM(1,0,q,e,2,NUM_QUAD)] = -J12 * r_idetJ;
         invJ[ijklNM(0,1,q,e,2,NUM_QUAD)] = -J21 * r_idetJ;
         invJ[ijklNM(1,1,q,e,2,NUM_QUAD)] =  J11 * r_idetJ;
         detJ[ijN(q,e,NUM_QUAD)] = r_detJ;
      }
   });
}

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> static
void Geom3D(const int numElements,
            const double* __restrict dofToQuadD,
            const double* __restrict nodes,
            double* __restrict J,
            double* __restrict invJ,
            double* __restrict detJ)
{
   const int NUM_DOFS = NUM_DOFS_1D*NUM_DOFS_1D*NUM_DOFS_1D;
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   MFEM_FORALL(e,numElements,
   {
      double s_nodes[3 * NUM_DOFS_1D * NUM_DOFS_1D * NUM_DOFS_1D];
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         for (int d = q; d < NUM_DOFS; d += NUM_QUAD)
         {
            s_nodes[ijN(0,d,3)] = nodes[ijkNM(0,d,e,3,NUM_DOFS)];
            s_nodes[ijN(1,d,3)] = nodes[ijkNM(1,d,e,3,NUM_DOFS)];
            s_nodes[ijN(2,d,3)] = nodes[ijkNM(2,d,e,3,NUM_DOFS)];
         }
      }
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double J11 = 0; double J12 = 0; double J13 = 0;
         double J21 = 0; double J22 = 0; double J23 = 0;
         double J31 = 0; double J32 = 0; double J33 = 0;
         for (int d = 0; d < NUM_DOFS; ++d)
         {
            const double wx = dofToQuadD[ijkNM(0,q,d,3,NUM_QUAD)];
            const double wy = dofToQuadD[ijkNM(1,q,d,3,NUM_QUAD)];
            const double wz = dofToQuadD[ijkNM(2,q,d,3,NUM_QUAD)];
            const double x = s_nodes[ijN(0,d,3)];
            const double y = s_nodes[ijN(1,d,3)];
            const double z = s_nodes[ijN(2,d,3)];
            J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
            J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
            J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
         }
         const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                                (J13 * J21 * J32) - (J13 * J22 * J31) -
                                (J12 * J21 * J33) - (J11 * J23 * J32));
         assert(r_detJ!=0.0);
         J[ijklNM(0,0,q,e,3,NUM_QUAD)] = J11;
         J[ijklNM(1,0,q,e,3,NUM_QUAD)] = J12;
         J[ijklNM(2,0,q,e,3,NUM_QUAD)] = J13;
         J[ijklNM(0,1,q,e,3,NUM_QUAD)] = J21;
         J[ijklNM(1,1,q,e,3,NUM_QUAD)] = J22;
         J[ijklNM(2,1,q,e,3,NUM_QUAD)] = J23;
         J[ijklNM(0,2,q,e,3,NUM_QUAD)] = J31;
         J[ijklNM(1,2,q,e,3,NUM_QUAD)] = J32;
         J[ijklNM(2,2,q,e,3,NUM_QUAD)] = J33;

         const double r_idetJ = 1.0 / r_detJ;
         invJ[ijklNM(0,0,q,e,3,NUM_QUAD)] = r_idetJ * ((J22 * J33)-(J23 * J32));
         invJ[ijklNM(1,0,q,e,3,NUM_QUAD)] = r_idetJ * ((J32 * J13)-(J33 * J12));
         invJ[ijklNM(2,0,q,e,3,NUM_QUAD)] = r_idetJ * ((J12 * J23)-(J13 * J22));
         invJ[ijklNM(0,1,q,e,3,NUM_QUAD)] = r_idetJ * ((J23 * J31)-(J21 * J33));
         invJ[ijklNM(1,1,q,e,3,NUM_QUAD)] = r_idetJ * ((J33 * J11)-(J31 * J13));
         invJ[ijklNM(2,1,q,e,3,NUM_QUAD)] = r_idetJ * ((J13 * J21)-(J11 * J23));
         invJ[ijklNM(0,2,q,e,3,NUM_QUAD)] = r_idetJ * ((J21 * J32)-(J22 * J31));
         invJ[ijklNM(1,2,q,e,3,NUM_QUAD)] = r_idetJ * ((J31 * J12)-(J32 * J11));
         invJ[ijklNM(2,2,q,e,3,NUM_QUAD)] = r_idetJ * ((J11 * J22)-(J12 * J21));
         detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
      }
   });
}

// *****************************************************************************
typedef void (*fIniGeom)(const int,const double*,const double*,
                         double*, double*, double*);

// *****************************************************************************
void Geom(const int DIM,
          const int NUM_DOFS,
          const int NUM_QUAD,
          const int numElements,
          const double* __restrict dofToQuad,
          const double* __restrict dofToQuadD,
          const double* __restrict nodes,
          double* __restrict x,
          double* __restrict J,
          double* __restrict invJ,
          double* __restrict detJ)
{
   const unsigned int dofs1D = IROOT(DIM,NUM_DOFS);
   const unsigned int quad1D = IROOT(DIM,NUM_QUAD);
   const unsigned int id = (DIM<<8)|(dofs1D-2)<<4|(quad1D-2);
   assert(LOG2(DIM)<=4);
   assert(LOG2(dofs1D-2)<=4);
   assert(LOG2(quad1D-2)<=4);
   static std::unordered_map<unsigned int, fIniGeom> call =
   {
      // 2D
      {0x200,&Geom2D<2,2>},
      {0x201,&Geom2D<2,3>},
      {0x202,&Geom2D<2,4>},
      {0x203,&Geom2D<2,5>},
      {0x204,&Geom2D<2,6>},
      {0x205,&Geom2D<2,7>},
      {0x206,&Geom2D<2,8>},
      {0x207,&Geom2D<2,9>},
      {0x210,&Geom2D<3,2>},
      {0x211,&Geom2D<3,3>},
      {0x212,&Geom2D<3,4>},
/*
      {0x208,&Geom2D<2,10>},
      {0x209,&Geom2D<2,11>},
      {0x20A,&Geom2D<2,12>},
      {0x20B,&Geom2D<2,13>},
      {0x20C,&Geom2D<2,14>},
      {0x20D,&Geom2D<2,15>},
      {0x20E,&Geom2D<2,16>},
      {0x20F,&Geom2D<2,17>},
*/
      // 3D
      {0x300,&Geom3D<2,2>},
      {0x301,&Geom3D<2,3>},
      {0x302,&Geom3D<2,4>},
      {0x303,&Geom3D<2,5>},
      {0x304,&Geom3D<2,6>},
      {0x305,&Geom3D<2,7>},
      {0x306,&Geom3D<2,8>},
      {0x307,&Geom3D<2,9>},
      {0x321,&Geom3D<4,3>},/*
      {0x308,&Geom3D<2,10>},
      {0x309,&Geom3D<2,11>},
      {0x30A,&Geom3D<2,12>},
      {0x30B,&Geom3D<2,13>},
      {0x30C,&Geom3D<2,14>},
      {0x30D,&Geom3D<2,15>},
      {0x30E,&Geom3D<2,16>},
      {0x30F,&Geom3D<2,17>},*/
   };
   if (!call[id])
   {
      printf("\n[Geom] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);
   GET_CONST_PTR(dofToQuadD);
   GET_CONST_PTR(nodes);
   GET_PTR(J);
   GET_PTR(invJ);
   GET_PTR(detJ);
   call[id](numElements, d_dofToQuadD, d_nodes, d_J, d_invJ, d_detJ);
}

} // namespace fem
} // namespace kernels
} // namespace mfem
