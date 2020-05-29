// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tmop.hpp"
#include "linearform.hpp"
#include "pgridfunc.hpp"
#include "tmop_tools.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void AddMultPA_Kernel_3D(const int NE,
                                const Array<double> &w_,
                                const Array<double> &b_,
                                const Array<double> &g_,
                                const Vector &d_,
                                const Vector &x_,
                                Vector &y_,
                                const int d1d = 0,
                                const int q1d = 0)
{

}

// *****************************************************************************
void TMOP_Integrator::AddMultPA_3D(const Vector &X, Vector &Y) const
{
   MFEM_VERIFY(IntRule,"");
   const int D1D = maps->ndof;
   const int Q1D = maps->nqpt;
   const IntegrationRule *ir = IntRule;
   const Array<double> &W = ir->GetWeights();
   const Array<double> &B1d = maps->B;
   const Array<double> &G1d = maps->G;
   const int id = (D1D << 4 ) | Q1D;

   // Jtr setup:
   //  - TargetConstructor::target_type == IDEAL_SHAPE_UNIT_SIZE
   //  - Jtr(i) == Wideal
   DenseMatrix Wideal(dim);
   static bool RAND = getenv("RAND");
   if (!RAND)
   {
      const FiniteElement *fe = fes->GetFE(0);
      const Geometry::Type geom_type = fe->GetGeomType();
      Wideal = Geometries.GetGeomToPerfGeomJac(geom_type);
      MFEM_VERIFY(Wideal.Det() == 1.0 ,"");
      {
         MFEM_VERIFY(Wideal(0,0)==1.0 && Wideal(1,1)==1.0 &&
                     Wideal(1,0)==0.0 && Wideal(0,1)==0.0,"");
      }
   }
   else
   {
      Wideal(0,0) = 1.0;
      Wideal(0,1) = 0.123;
      Wideal(1,0) = 0.456;
      Wideal(1,1) = 1.0;
   }
   /*
      Array<int> vdofs;
      DenseTensor Jtr(dim, dim, ir->GetNPoints());
      for (int i = 0; i < fes->GetNE(); i++)
      {
         const FiniteElement *el = fes->GetFE(i);
         fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         px.GetSubVector(vdofs, el_x);
         targetC->ComputeElementTargets(T.ElementNo, el, *ir, elfun, Jtr);
     }*/
   const auto Jtr = Reshape(Wideal.Read(), dim, dim);
   auto J = Reshape(Dpa.Write(), Q1D, Q1D, dim, dim, ne);
   MFEM_FORALL_2D(e, ne, Q1D, Q1D, 1,
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            J(qx,qy,0,0,e) = Jtr(0,0);
            J(qx,qy,0,1,e) = Jtr(0,1);
            J(qx,qy,1,0,e) = Jtr(1,0);
            J(qx,qy,1,1,e) = Jtr(1,1);
         }
      }
   });

   switch (id)
   {
      case 0x21: return AddMultPA_Kernel_3D<2,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x22: return AddMultPA_Kernel_3D<2,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x23: return AddMultPA_Kernel_3D<2,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x24: return AddMultPA_Kernel_3D<2,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x25: return AddMultPA_Kernel_3D<2,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x26: return AddMultPA_Kernel_3D<2,6,1>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x31: return AddMultPA_Kernel_3D<3,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x32: return AddMultPA_Kernel_3D<3,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x33: return AddMultPA_Kernel_3D<3,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x34: return AddMultPA_Kernel_3D<3,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x35: return AddMultPA_Kernel_3D<3,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x36: return AddMultPA_Kernel_3D<3,6,1>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x41: return AddMultPA_Kernel_3D<4,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x42: return AddMultPA_Kernel_3D<4,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x43: return AddMultPA_Kernel_3D<4,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x44: return AddMultPA_Kernel_3D<4,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x45: return AddMultPA_Kernel_3D<4,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x46: return AddMultPA_Kernel_3D<4,6,1>(ne,W,B1d,G1d,Dpa,X,Y);

      case 0x51: return AddMultPA_Kernel_3D<5,1,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x52: return AddMultPA_Kernel_3D<5,2,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x53: return AddMultPA_Kernel_3D<5,3,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x54: return AddMultPA_Kernel_3D<5,4,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x55: return AddMultPA_Kernel_3D<5,5,1>(ne,W,B1d,G1d,Dpa,X,Y);
      case 0x56: return AddMultPA_Kernel_3D<5,6,1>(ne,W,B1d,G1d,Dpa,X,Y);
      default:  break;
   }
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
