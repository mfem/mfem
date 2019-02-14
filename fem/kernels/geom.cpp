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
static void Geom2D(const int NUM_DOFS,
                   const int NUM_QUAD,
                   const int ne,
                   const double* __restrict B,
                   const double* __restrict dofToQuadD,
                   const double* __restrict nodes,
                   double* __restrict x,
                   double* __restrict J,
                   double* __restrict invJ,
                   double* __restrict detJ)
{
   // number of doubles in shared memory per threads
   const int Nspt = 2*NUM_DOFS;
   MFEM_FORALL_SHARED(e, ne, Nspt,
   {
      double *s_nodes = __shared;
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
static void Geom3D(const int NUM_DOFS,
                   const int NUM_QUAD,
                   const int ne,
                   const double* __restrict B,
                   const double* __restrict dofToQuadD,
                   const double* __restrict nodes,
                   double* __restrict x,
                   double* __restrict J,
                   double* __restrict invJ,
                   double* __restrict detJ)
{
   // number of doubles in shared memory per threads
   const int Nspt = 3 * NUM_DOFS;
   MFEM_FORALL_SHARED(e, ne, Nspt,
   {
      double *s_nodes = __shared;
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
void Geom(const int DIM,
          const int NUM_DOFS,
          const int NUM_QUAD,
          const int numElements,
          const double* __restrict B,
          const double* __restrict G,
          const double* __restrict X,
          double* __restrict x,
          double* __restrict J,
          double* __restrict invJ,
          double* __restrict detJ)
{
   GET_CONST_PTR(B);
   GET_CONST_PTR(G);
   GET_CONST_PTR(X);
   GET_PTR(x);
   GET_PTR(J);
   GET_PTR(invJ);
   GET_PTR(detJ);
   
   if (DIM==2){
      return Geom2D(NUM_DOFS, NUM_QUAD,
                    numElements, d_B, d_G, d_X, d_x, d_J, d_invJ, d_detJ);
   }
   
   if (DIM==2){
      return Geom3D(NUM_DOFS, NUM_QUAD,
                    numElements, d_B, d_G, d_X, d_x, d_J, d_invJ, d_detJ);
   }
}

} // namespace fem
} // namespace kernels
} // namespace mfem
