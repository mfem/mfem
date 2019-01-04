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

#include "../../general/okina.hpp"

namespace mfem
{

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
void kGeom2D(const int numElements,
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
template void kGeom2D<2,2>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,3>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,4>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,5>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,6>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,7>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,8>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,9>(int, double const*, double const*, double*, double*,
                           double*);
template void kGeom2D<2,10>(int, double const*, double const*, double*, double*,
                            double*);
template void kGeom2D<2,11>(int, double const*, double const*, double*, double*,
                            double*);
template void kGeom2D<2,12>(int, double const*, double const*, double*, double*,
                            double*);
template void kGeom2D<2,13>(int, double const*, double const*, double*, double*,
                            double*);
template void kGeom2D<2,14>(int, double const*, double const*, double*, double*,
                            double*);
template void kGeom2D<2,15>(int, double const*, double const*, double*, double*,
                            double*);
template void kGeom2D<2,16>(int, double const*, double const*, double*, double*,
                            double*);
template void kGeom2D<2,17>(int, double const*, double const*, double*, double*,
                            double*);

template void kGeom2D<3,2>(int, double const*, double const*, double*, double*,
                            double*);
}
