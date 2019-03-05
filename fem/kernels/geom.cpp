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
#include "../../linalg/device.hpp"

namespace mfem
{
namespace kernels
{
namespace fem
{

// *****************************************************************************
template<const int ND1d,
         const int NQ1d> static
void Geom2D(const int NE,
            const double* __restrict _G,
            const double* __restrict _X,
            double* __restrict _Xq,
            double* __restrict _J,
            double* __restrict _invJ,
            double* __restrict _detJ)
{
   const int ND = ND1d*ND1d;
   const int NQ = NQ1d*NQ1d;
   const DeviceTensor<3> G(_G, 2,NQ,ND);
   const DeviceTensor<3> X(_X, 2,ND,NE);
   DeviceTensor<3> Xq(_Xq, 2,NQ,NE);
   DeviceTensor<4> J(_J, 2,2,NQ,NE);
   DeviceTensor<4> invJ(_invJ, 2,2,NQ,NE);
   DeviceMatrix detJ(_detJ, NQ,NE);
   MFEM_FORALL(e, NE,
   {
      double s_X[2*ND1d*ND1d];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d +=NQ)
         {
            s_X[0+d*2] = X(0,d,e);
            s_X[1+d*2] = X(1,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double J11 = 0; double J12 = 0;
         double J21 = 0; double J22 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double x = s_X[0+d*2];
            const double y = s_X[1+d*2];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
         }
         const double r_detJ = (J11 * J22)-(J12 * J21);
         assert(r_detJ!=0.0);
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) =  J22 * r_idetJ;
         invJ(1,0,q,e) = -J12 * r_idetJ;
         invJ(0,1,q,e) = -J21 * r_idetJ;
         invJ(1,1,q,e) =  J11 * r_idetJ;
         detJ(q,e) = r_detJ;
      }
   });
}

// *****************************************************************************
template<const int ND1d,
         const int NQ1d> static
void Geom3D(const int NE,
            const double* __restrict _G,
            const double* __restrict _X,
            double* __restrict _Xq,
            double* __restrict _J,
            double* __restrict _invJ,
            double* __restrict _detJ)
{
   const int ND = ND1d*ND1d*ND1d;
   const int NQ = NQ1d*NQ1d*NQ1d;
   const DeviceTensor<3> G(_G, 3,NQ,NE);
   const DeviceTensor<3> X(_X, 3,ND,NE);
   DeviceTensor<3> Xq(_Xq, 3,NQ,NE);
   DeviceTensor<4> J(_J, 3,3,NQ,NE);
   DeviceTensor<4> invJ(_invJ, 3,3,NQ,NE);
   DeviceMatrix detJ(_detJ, NQ,NE);
   MFEM_FORALL(e,NE,
   {
      double s_nodes[3*ND1d*ND1d*ND1d];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d += NQ)
         {
            s_nodes[0+d*3] = X(0,d,e);
            s_nodes[1+d*3] = X(1,d,e);
            s_nodes[2+d*3] = X(2,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double J11 = 0; double J12 = 0; double J13 = 0;
         double J21 = 0; double J22 = 0; double J23 = 0;
         double J31 = 0; double J32 = 0; double J33 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double wz = G(2,q,d);
            const double x = s_nodes[0+d*3];
            const double y = s_nodes[1+d*3];
            const double z = s_nodes[2+d*3];
            J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
            J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
            J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
         }
         const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                                (J13 * J21 * J32) - (J13 * J22 * J31) -
                                (J12 * J21 * J33) - (J11 * J23 * J32));
         assert(r_detJ!=0.0);
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(2,0,q,e) = J13;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         J(2,1,q,e) = J23;
         J(0,2,q,e) = J31;
         J(1,2,q,e) = J32;
         J(2,2,q,e) = J33;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
         invJ(1,0,q,e) = r_idetJ * ((J32 * J13)-(J33 * J12));
         invJ(2,0,q,e) = r_idetJ * ((J12 * J23)-(J13 * J22));
         invJ(0,1,q,e) = r_idetJ * ((J23 * J31)-(J21 * J33));
         invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
         invJ(2,1,q,e) = r_idetJ * ((J13 * J21)-(J11 * J23));
         invJ(0,2,q,e) = r_idetJ * ((J21 * J32)-(J22 * J31));
         invJ(1,2,q,e) = r_idetJ * ((J31 * J12)-(J32 * J11));
         invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
         detJ(q,e) = r_detJ;
      }
   });
}

// *****************************************************************************
typedef void (*fIniGeom)(const int ne,
                         const double *G, const double *X,
                         double *x, double *J, double *invJ, double *detJ);

// *****************************************************************************
void Geom(const int dim,
          const int ND,
          const int NQ,
          const int NE,
          const double* __restrict G,
          const double* __restrict X,
          double* __restrict Xq,
          double* __restrict J,
          double* __restrict invJ,
          double* __restrict detJ)
{
   const unsigned int ND1d = IROOT(dim,ND);
   const unsigned int NQ1d = IROOT(dim,NQ);
   const unsigned int id = (dim<<8)|(ND1d)<<4|(NQ1d);
   assert(LOG2(dim)<=4);
   assert(LOG2(ND1d)<=4);
   assert(LOG2(NQ1d)<=4);
   static std::unordered_map<unsigned int, fIniGeom> call =
   {
      {0x222,&Geom2D<2,2>},
      {0x224,&Geom2D<2,4>},
      {0x232,&Geom2D<3,2>},
      {0x242,&Geom2D<4,2>},
      {0x234,&Geom2D<3,4>},
      {0x323,&Geom3D<2,3>},
      {0x334,&Geom3D<3,4>},
   };
   if (!call[id])
   {
      printf("dim=%d, ND1d=%d and NQ1d=%d",dim, ND1d, NQ1d);
      mfem_error("Geom kernel not instanciated");
   }
   call[id](NE, G, X, Xq, J, invJ, detJ);
}

} // namespace fem
} // namespace kernels
} // namespace mfem
