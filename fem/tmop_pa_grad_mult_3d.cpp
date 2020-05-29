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
#define MFEM_DBG_COLOR 211
#include "../general/dbg.hpp"
#include "../general/forall.hpp"
#include "../linalg/kernels.hpp"
#include "../linalg/dtensor.hpp"

namespace mfem
{

// *****************************************************************************
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void AddMultGradPA_Kernel_3D(const int NE,
                                    const Array<double> &b1d_,
                                    const Array<double> &g1d_,
                                    const DenseMatrix &Jtr,
                                    const Vector &p_,
                                    const Vector &x_,
                                    Vector &y_,
                                    const int d1d = 0,
                                    const int q1d = 0)
{
}

// *****************************************************************************
void TMOP_Integrator::AddMultGradPA_3D(const Vector &Xe, const Vector &Re,
                                       Vector &Ce) const
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
   // Get Wideal into Jtr
   DenseMatrix Jtr(dim);
   static bool RAND = getenv("RAND");
   if (!RAND)
   {
      const FiniteElement *fe = fes->GetFE(0);
      const Geometry::Type geom_type = fe->GetGeomType();
      Jtr = Geometries.GetGeomToPerfGeomJac(geom_type);
      MFEM_VERIFY(Jtr.Det() == 1.0 ,"");
      {
         MFEM_VERIFY(Jtr(0,0)==1.0 && Jtr(1,1)==1.0 &&
                     Jtr(1,0)==0.0 && Jtr(0,1)==0.0,"");
      }
   }
   else
   {
      Jtr(0,0) = 1.0;
      Jtr(0,1) = 0.123;
      Jtr(1,0) = 0.456;
      Jtr(1,1) = 1.0;
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
   if (!setup)
   {
      setup = true;
      AssembleGradPA_3D(Jtr,Xe);
   }

   switch (id)
   {
      case 0x21: return AddMultGradPA_Kernel_3D<2,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x22: return AddMultGradPA_Kernel_3D<2,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x23: return AddMultGradPA_Kernel_3D<2,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x24: return AddMultGradPA_Kernel_3D<2,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x25: return AddMultGradPA_Kernel_3D<2,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x26: return AddMultGradPA_Kernel_3D<2,6,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);

      case 0x31: return AddMultGradPA_Kernel_3D<3,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x32: return AddMultGradPA_Kernel_3D<3,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x33: return AddMultGradPA_Kernel_3D<3,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x34: return AddMultGradPA_Kernel_3D<3,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x35: return AddMultGradPA_Kernel_3D<3,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x36: return AddMultGradPA_Kernel_3D<3,6,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);

      case 0x41: return AddMultGradPA_Kernel_3D<4,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x42: return AddMultGradPA_Kernel_3D<4,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x43: return AddMultGradPA_Kernel_3D<4,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x44: return AddMultGradPA_Kernel_3D<4,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x45: return AddMultGradPA_Kernel_3D<4,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x46: return AddMultGradPA_Kernel_3D<4,6,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);

      case 0x51: return AddMultGradPA_Kernel_3D<5,1,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x52: return AddMultGradPA_Kernel_3D<5,2,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x53: return AddMultGradPA_Kernel_3D<5,3,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x54: return AddMultGradPA_Kernel_3D<5,4,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x55: return AddMultGradPA_Kernel_3D<5,5,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      case 0x56: return AddMultGradPA_Kernel_3D<5,6,1>(ne,B1d,G1d,Jtr,dPpa,Re,Ce);
      default:  break;
   }
   dbg("kernel id: %x", id);
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem
